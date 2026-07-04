"""NeuroScale++ Main Entry Point.

Usage:
    # Full pipeline (all 3 phases + evaluation + plots)
    python main.py --config configs/cifar10.yaml --mode all

    # Individual phases
    python main.py --config configs/cifar10.yaml --mode phase1
    python main.py --config configs/cifar10.yaml --mode phase2
    python main.py --config configs/cifar10.yaml --mode phase3
    python main.py --config configs/cifar10.yaml --mode evaluate

    # Resume training from checkpoint
    python main.py --config configs/cifar10.yaml --mode phase1 --resume checkpoints/ann_epoch100.pth

    # Regenerate plots from existing results
    python main.py --config configs/cifar10.yaml --mode plots --results-dir results/cifar10_20260624_120000

All results (CSVs, JSONs, plots) are saved to:
    results/<dataset>_<timestamp>/
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from neuroscale.utils.config import load_config
from neuroscale.utils.logging import setup_logger
from neuroscale.utils.results_manager import ResultsManager
from neuroscale.utils.plotting import (
    generate_all_plots,
    generate_phase2_plots_from_file,
    generate_eval_plots_from_csv,
    plot_phase2_difficulty_histogram,
    plot_phase2_scaling_law_samples,
    plot_phase2_parameter_distributions,
    plot_confidence_histogram,
    plot_timestep_vs_confidence,
    plot_per_class_accuracy,
)
from neuroscale.datasets.factory import get_dataloaders
from neuroscale.models.factory import get_ann_model
from neuroscale.training.phase1_ann import train_ann
from neuroscale.training.phase2_profiling import profile_snn, load_profiling_results
from neuroscale.training.phase3_joint import train_joint, load_phase3_checkpoint
from neuroscale.conversion.converter import ANNtoSNNConverter
from neuroscale.scaling.curve_fitting import compute_optimal_timestep
from neuroscale.spiking.snn_model import SNNModel
from neuroscale.spiking.multi_exit_snn import MultiExitSNN
from neuroscale.predictor.complexity_predictor import ComplexityPredictor
from neuroscale.inference.adaptive_inference import AdaptiveInference


def parse_args():
    parser = argparse.ArgumentParser(
        description="NeuroScale++: Energy-Efficient ANN-to-SNN Conversion"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to config YAML (e.g., configs/cifar10.yaml)"
    )
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["all", "phase1", "phase2", "phase3", "evaluate", "plots"],
        help="Which phase to run (default: all)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from (Phase 1)"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="./checkpoints",
        help="Directory to save/load checkpoints"
    )
    parser.add_argument(
        "--log-dir", type=str, default="./logs",
        help="Directory for TensorBoard logs"
    )
    parser.add_argument(
        "--results-dir", type=str, default="./results",
        help="Base directory for organized results output"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (e.g., 'cuda', 'cuda:0', 'mps', 'cpu'). Auto-detected if not set."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str: str = None) -> torch.device:
    """Determine computation device."""
    if device_str is not None:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_phase1(config: dict, device: torch.device, args, rm: ResultsManager) -> torch.nn.Module:
    """Run Phase 1: ANN Pretraining."""
    print("\n" + "=" * 60)
    print("PHASE 1: ANN Pretraining")
    print("=" * 60)

    model = train_ann(
        config=config,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        resume_from=args.resume,
        results_manager=rm,
    )
    return model


def run_phase2(config: dict, device: torch.device, args, rm: ResultsManager) -> dict:
    """Run Phase 2: SNN Profiling."""
    print("\n" + "=" * 60)
    print("PHASE 2: SNN Profiling & Curve Fitting")
    print("=" * 60)

    # Load trained ANN
    ann_model = get_ann_model(config).to(device)
    ann_ckpt_path = Path(args.checkpoint_dir) / "ann_best.pth"

    if not ann_ckpt_path.exists():
        raise FileNotFoundError(
            f"ANN checkpoint not found at {ann_ckpt_path}. Run Phase 1 first."
        )

    ckpt = torch.load(ann_ckpt_path, map_location=device)
    ann_model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded ANN from {ann_ckpt_path} (acc={ckpt.get('best_acc', '?')}%)")

    # Get training data
    train_loader, _ = get_dataloaders(config)

    # Run profiling
    results = profile_snn(
        ann_model=ann_model,
        train_loader=train_loader,
        config=config,
        device=device,
        save_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        results_manager=rm,
    )

    # Generate Phase 2 specific plots immediately
    plots_dir = rm.get_plots_dir()
    alphas, betas, gammas = results["alphas"], results["betas"], results["gammas"]
    target_acc = config["scaling_law"]["target_accuracy"]
    max_t = config["snn"]["max_timestep"]
    t_optimals = np.array([
        compute_optimal_timestep(a, b, g, target_acc, max_timestep=max_t)
        for a, b, g in zip(alphas, betas, gammas)
    ])

    plot_phase2_difficulty_histogram(t_optimals, config["snn"]["exit_points"], plots_dir)
    plot_phase2_scaling_law_samples(
        results["timesteps"], results["performances"],
        alphas, betas, gammas, plots_dir
    )
    plot_phase2_parameter_distributions(alphas, betas, gammas, plots_dir)
    print(f"  Phase 2 plots saved to {plots_dir}")

    return results


def run_phase3(config: dict, device: torch.device, args, rm: ResultsManager) -> tuple:
    """Run Phase 3: Joint Training."""
    print("\n" + "=" * 60)
    print("PHASE 3: Joint Training (Predictor + Exit Branches)")
    print("=" * 60)

    # Load converted SNN
    snn_ckpt_path = Path(args.checkpoint_dir) / "snn_converted.pth"
    if not snn_ckpt_path.exists():
        raise FileNotFoundError(
            f"SNN checkpoint not found at {snn_ckpt_path}. Run Phase 2 first."
        )

    snn_ckpt = torch.load(snn_ckpt_path, map_location=device)
    thresholds = snn_ckpt["thresholds"]

    # Rebuild SNN model from ANN architecture
    ann_model = get_ann_model(config).to(device)
    ann_ckpt = torch.load(
        Path(args.checkpoint_dir) / "ann_best.pth", map_location=device
    )
    ann_model.load_state_dict(ann_ckpt["model_state_dict"])

    # Convert again (needed to get properly normalized weights)
    converter = ANNtoSNNConverter(
        percentile=config["conversion"].get("percentile", 99.9),
        calibrate=False,
    )
    train_loader, _ = get_dataloaders(config)

    from neuroscale.training.phase2_profiling import _make_calibration_loader
    calib_loader = _make_calibration_loader(train_loader, config)
    snn_base, _ = converter.convert(ann_model, calib_loader, device)

    snn_model = SNNModel(
        ann_model=snn_base,
        thresholds=thresholds,
        max_timestep=config["snn"]["max_timestep"],
        neuron_type="if",
    ).to(device)

    # Load SNN state dict
    snn_model.load_state_dict(snn_ckpt["snn_state_dict"])

    # Load profiling results
    profiling_results = load_profiling_results(args.checkpoint_dir)
    print(f"Loaded profiling: {len(profiling_results['alphas'])} samples")

    # Run joint training
    multi_exit_snn, predictor = train_joint(
        snn_model=snn_model,
        train_loader=train_loader,
        profiling_results=profiling_results,
        config=config,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        results_manager=rm,
    )

    return multi_exit_snn, predictor


def run_evaluate(config: dict, device: torch.device, args, rm: ResultsManager) -> dict:
    """Run evaluation with adaptive inference + full metrics + plots."""
    print("\n" + "=" * 60)
    print("EVALUATION: Adaptive Inference")
    print("=" * 60)

    # Rebuild full system from checkpoints
    phase3_path = Path(args.checkpoint_dir) / "phase3_best.pth"
    snn_path = Path(args.checkpoint_dir) / "snn_converted.pth"

    if not phase3_path.exists():
        raise FileNotFoundError(
            f"Phase 3 checkpoint not found at {phase3_path}. Run Phase 3 first."
        )

    # Load SNN
    snn_ckpt = torch.load(snn_path, map_location=device)
    thresholds = snn_ckpt["thresholds"]

    ann_model = get_ann_model(config).to(device)
    ann_ckpt = torch.load(
        Path(args.checkpoint_dir) / "ann_best.pth", map_location=device
    )
    ann_model.load_state_dict(ann_ckpt["model_state_dict"])

    # Rebuild SNN
    converter = ANNtoSNNConverter(
        percentile=config["conversion"].get("percentile", 99.9),
        calibrate=False,
    )
    train_loader, test_loader = get_dataloaders(config)

    from neuroscale.training.phase2_profiling import _make_calibration_loader
    calib_loader = _make_calibration_loader(train_loader, config)
    snn_base, _ = converter.convert(ann_model, calib_loader, device)

    snn_model = SNNModel(
        ann_model=snn_base,
        thresholds=thresholds,
        max_timestep=config["snn"]["max_timestep"],
        neuron_type="if",
    ).to(device)
    snn_model.load_state_dict(snn_ckpt["snn_state_dict"])

    # Build Multi-Exit SNN
    multi_exit_snn = MultiExitSNN(
        snn_model=snn_model,
        exit_timesteps=config["snn"]["exit_points"],
        num_classes=config["dataset"]["num_classes"],
        hidden_dim=128,
    ).to(device)

    # Build predictor
    predictor = ComplexityPredictor(
        in_channels=3,
        image_size=config["dataset"]["image_size"],
        hidden_dims=config["predictor"]["hidden_dims"],
    ).to(device)

    # Load Phase 3 weights
    load_phase3_checkpoint(str(phase3_path), multi_exit_snn, predictor, device)
    print("Loaded Phase 3 checkpoint (exit branches + predictor)")

    # Run full evaluation with ResultsManager integration
    inference = AdaptiveInference(multi_exit_snn, predictor, config)
    num_classes = config["dataset"]["num_classes"]

    # Load Phase 2 profiling results for predictor T-error computation
    from neuroscale.training.phase2_profiling import load_profiling_results as _load_prof
    profiling_results = None
    try:
        profiling_results = _load_prof(args.checkpoint_dir)
    except FileNotFoundError:
        pass  # Phase 2 results optional for T-error metric

    results = inference.evaluate_full(
        test_loader=test_loader,
        device=device,
        results_manager=rm,
        num_classes=num_classes,
        profiling_results=profiling_results,
    )

    # Generate evaluation-specific plots
    plots_dir = rm.get_plots_dir()
    predictions = results["_predictions"]
    targets = results["_targets"]
    confidences = results["_confidences"]
    timesteps_used = results["_timesteps_used"]
    correct = (predictions == targets).astype(float)

    plot_confidence_histogram(confidences, correct, plots_dir)
    plot_timestep_vs_confidence(timesteps_used, confidences, correct, plots_dir)

    # Per-class analysis
    per_class_acc = np.zeros(num_classes)
    per_class_avg_t = np.zeros(num_classes)
    for c in range(num_classes):
        mask = targets == c
        if mask.sum() > 0:
            per_class_acc[c] = (predictions[mask] == targets[mask]).mean() * 100
            per_class_avg_t[c] = timesteps_used[mask].mean()
    plot_per_class_accuracy(per_class_acc, per_class_avg_t, plots_dir)

    print(f"\n  Evaluation plots saved to {plots_dir}")

    return results


def run_plots_only(config: dict, args, rm: ResultsManager):
    """Regenerate all plots from existing results (no training)."""
    print("\n" + "=" * 60)
    print("REGENERATING PLOTS")
    print("=" * 60)

    # Generate standard plots from ResultsManager data
    generate_all_plots(rm, config)

    # Also generate Phase 2 plots from saved profiling file if available
    profiling_path = Path(args.checkpoint_dir) / "profiling_results.npz"
    if profiling_path.exists():
        generate_phase2_plots_from_file(
            str(profiling_path), rm.get_plots_dir(), config
        )

    # Generate eval plots from CSVs if available
    eval_dir = rm.eval_dir
    if (eval_dir / "adaptive_results.csv").exists():
        generate_eval_plots_from_csv(eval_dir, rm.get_plots_dir(), config)


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)
    dataset_name = config["dataset"]["name"]
    print(f"NeuroScale++ | Dataset: {dataset_name} | Mode: {args.mode}")

    # Setup
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}")

    # Create output directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # Initialize ResultsManager
    rm = ResultsManager(config, results_dir=args.results_dir)
    print(f"Results: {rm.base_dir}")

    # Run selected mode
    if args.mode == "all":
        run_phase1(config, device, args, rm)
        run_phase2(config, device, args, rm)
        run_phase3(config, device, args, rm)
        run_evaluate(config, device, args, rm)

        # Generate master plot set at the end
        print("\n" + "=" * 60)
        print("GENERATING ALL PLOTS & FINAL REPORTS")
        print("=" * 60)
        generate_all_plots(rm, config)

    elif args.mode == "phase1":
        run_phase1(config, device, args, rm)
        generate_all_plots(rm, config)

    elif args.mode == "phase2":
        run_phase2(config, device, args, rm)
        generate_all_plots(rm, config)

    elif args.mode == "phase3":
        run_phase3(config, device, args, rm)
        generate_all_plots(rm, config)

    elif args.mode == "evaluate":
        run_evaluate(config, device, args, rm)
        generate_all_plots(rm, config)

    elif args.mode == "plots":
        run_plots_only(config, args, rm)

    # Finalize — close files, write experiment summary
    rm.finalize()

    print("\n" + "=" * 60)
    print("DONE!")
    print(f"  All results: {rm.base_dir}")
    print(f"  CSVs:        {rm.base_dir}/phase*/")
    print(f"  Plots:       {rm.get_plots_dir()}")
    print(f"  Summaries:   {rm.base_dir}/*/summary.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
