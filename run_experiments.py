"""
Fine-tune OLMo (QLoRA) on MoleculeNet datasets with multi-seed evaluation.

Each experiment is run 3 times with different seeds (configurable in experiments.yaml).
Training uses early stopping (patience 3). After training, only LoRA adapters and the
classifier head are saved. For evaluation, adapters are loaded onto a fresh base model
and the test set is evaluated via trainer.test().

Usage:
    python run_experiments.py --datasets bbbp
    python run_experiments.py --datasets bbbp bace_classification
    python run_experiments.py --datasets all
    python run_experiments.py --datasets bbbp --config my_config.yaml
    python run_experiments.py --datasets all --dry-run
    python run_experiments.py --datasets bbbp --strategy ddp --devices 4
"""

import argparse
import importlib
import json
import yaml
import torch
import numpy as np
import os
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

import Model
import dataset

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_config(config_path: str) -> dict:
    """Load YAML config and return the parsed dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_dataset_config(config: dict, dataset_name: str) -> dict:
    """Merge per-dataset config with defaults and return a complete config."""
    defaults = config.get("defaults", {})
    datasets_cfg = config.get("datasets", {})

    if dataset_name not in datasets_cfg:
        available = ", ".join(datasets_cfg.keys())
        raise ValueError(
            f"Dataset '{dataset_name}' not found in config. Available: {available}"
        )

    ds_cfg = datasets_cfg[dataset_name]
    # Merge: per-dataset values override defaults
    merged = {**defaults, **ds_cfg}
    return merged


# ---------------------------------------------------------------------------
# DeepChem loader resolver
# ---------------------------------------------------------------------------
def resolve_loader(loader_name: str):
    """
    Dynamically import a DeepChem MoleculeNet loader by name.

    Example: resolve_loader("load_bbbp") → deepchem.molnet.load_bbbp
    """
    try:
        molnet = importlib.import_module("deepchem.molnet")
        loader_fn = getattr(molnet, loader_name)
        return loader_fn
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Could not resolve loader '{loader_name}' from deepchem.molnet: {e}"
        )


# ---------------------------------------------------------------------------
# Single seed training + evaluation
# ---------------------------------------------------------------------------
def train_single_seed(
    ds_config: dict,
    dataset_name: str,
    seed: int,
    output_root: Path,
) -> dict:
    """
    Train the model for one seed, save adapters, reload them, and evaluate on test.

    Returns a dict with the test metric name and value.
    """
    # --- Seed everything ---
    pl.seed_everything(seed, workers=True)

    # --- Resolve the DeepChem loader ---
    loader_fn = resolve_loader(ds_config["loader"])

    # --- Extract hyperparameters ---
    model_name = ds_config["model_name"]
    base_model = ds_config["base_model"]
    task_type = ds_config["task_type"]
    metric = ds_config["metric"]
    metric_mode = ds_config["metric_mode"]
    max_epochs = ds_config["max_epochs"]
    patience = ds_config["early_stopping_patience"]
    precision = ds_config.get("precision", "bf16-mixed")
    batch_size = ds_config["batch_size"]
    eval_batch_size = ds_config.get("eval_batch_size", batch_size)
    strategy = ds_config.get("strategy", None)
    
    num_workers = os.cpu_count() or 4
    if strategy is None:
        pl_strategy = "auto"
        devices = 1
    elif strategy == "ddp":
        pl_strategy = "ddp"
        devices = "auto"
    else:
        pl_strategy = strategy
        devices = "auto"

    hparams_config = {
        "lr": ds_config["lr"],
        "lora_r": ds_config["lora_r"],
        "lora_alpha": ds_config["lora_alpha"],
        "lora_dropout": ds_config["lora_dropout"],
        "weight_decay": ds_config["weight_decay"],
    }

    # --- Output directory for this seed ---
    seed_dir = output_root / dataset_name / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = seed_dir / "adapters"

    # --- Create datasets ---
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name} | Seed: {seed}")
    print(f"{'='*60}")
    print(f"  Hyperparameters: {json.dumps(hparams_config, indent=2)}")

    train_ds = dataset.MolnetDataset(
        loader_fn, mode="train", model=model_name, do_random=True, prob=0.2
    )
    val_ds = dataset.MolnetDataset(
        loader_fn, mode="valid", model=model_name, do_random=False
    )
    test_ds = dataset.MolnetDataset(
        loader_fn, mode="test", model=model_name, do_random=False
    )

    num_tasks = train_ds.num_tasks
    print(f"  Num tasks: {num_tasks}")
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # --- DataLoaders ---
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # --- Model ---
    model = Model.OlmoModelQLora(
        hparams_config=hparams_config,
        num_tasks=num_tasks,
        task_type=task_type,
        model_name=model_name,
        base_model=base_model,
    )

    # --- Callbacks ---
    early_stopping = EarlyStopping(
        monitor=metric,
        patience=patience,
        mode=metric_mode,
        verbose=True,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(seed_dir / "checkpoints"),
        filename=f"{dataset_name}_seed{seed}_best",
        monitor=metric,
        mode=metric_mode,
        save_top_k=1,
        verbose=True,
    )

    # --- Trainer ---
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        precision=precision,
        accelerator="gpu",
        strategy=pl_strategy,
        devices=devices,
        callbacks=[early_stopping, checkpoint_callback],
        enable_progress_bar=True,
        log_every_n_steps=10,
        deterministic=False,  # some ops don't support deterministic on GPU
    )

    # --- Train ---
    print(f"\n  Starting training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"  Training complete. Best checkpoint: {checkpoint_callback.best_model_path}")

    # --- Save LoRA adapters and classifier head from best checkpoint ---
    best_model = Model.OlmoModelQLora.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        hparams_config=hparams_config,
        num_tasks=num_tasks,
        task_type=task_type,
        model_name=model_name,
        base_model=base_model,
    )

    # Save the LoRA adapters
    best_model.model.save_pretrained(str(adapter_dir))
    print(f"  LoRA adapters saved to: {adapter_dir}")

    # Save the classifier head
    classifier_path = seed_dir / "classifier.pt"
    torch.save(best_model.classifier.state_dict(), str(classifier_path))
    print(f"  Classifier head saved to: {classifier_path}")

    # --- Evaluation: load adapters onto a fresh model and test ---
    print(f"\n  Loading adapters for test evaluation...")

    # Create a fresh model that will load the saved adapters
    # We pass the adapter_dir as model_name so _load_model detects the adapter_config.json
    eval_hparams = hparams_config.copy()
    eval_model = Model.OlmoModelQLora(
        hparams_config=eval_hparams,
        num_tasks=num_tasks,
        task_type=task_type,
        model_name=str(adapter_dir),  # point to the saved adapters
        base_model=base_model,
    )
    # configure_model is called lazily by Lightning, but we need to ensure
    # the classifier head is loaded after model setup
    eval_model.configure_model()

    # Load the saved classifier head
    eval_model.classifier.load_state_dict(torch.load(str(classifier_path), weights_only=True))
    print(f"  Adapters and classifier loaded successfully.")

    # --- Run test ---
    eval_trainer = pl.Trainer(
        precision=precision,
        accelerator="gpu",
        strategy=pl_strategy,
        devices=devices,
        enable_progress_bar=True,
    )

    test_results = eval_trainer.test(eval_model, dataloaders=test_loader)

    # --- Extract test metric ---
    test_metric_name = metric.replace("val_", "test_")
    test_metric_value = test_results[0].get(test_metric_name, None)
    print(f"  Test metric ({test_metric_name}): {test_metric_value}")

    # --- Cleanup GPU memory ---
    del model, best_model, eval_model, trainer, eval_trainer
    torch.cuda.empty_cache()

    return {
        "seed": seed,
        "metric_name": test_metric_name,
        "metric_value": test_metric_value,
        "best_checkpoint": checkpoint_callback.best_model_path,
        "adapter_dir": str(adapter_dir),
    }


# ---------------------------------------------------------------------------
# Run all seeds for a dataset
# ---------------------------------------------------------------------------
def run_dataset(config: dict, dataset_name: str, output_root: Path) -> dict:
    """
    Run training + evaluation across all seeds for a single dataset.
    Returns aggregated results.
    """
    ds_config = get_dataset_config(config, dataset_name)
    seeds = ds_config.get("seeds", [42, 43, 44])

    print(f"\n{'#'*70}")
    print(f"  DATASET: {dataset_name}")
    print(f"  Seeds: {seeds}")
    print(f"{'#'*70}")

    all_results = []
    for seed in seeds:
        result = train_single_seed(ds_config, dataset_name, seed, output_root)
        all_results.append(result)

    # --- Aggregate results ---
    metric_name = all_results[0]["metric_name"]
    metric_values = [r["metric_value"] for r in all_results if r["metric_value"] is not None]

    mean_metric = np.mean(metric_values) if metric_values else float("nan")
    std_metric = np.std(metric_values) if metric_values else float("nan")

    print(f"\n{'='*60}")
    print(f"  === {dataset_name} Results ({len(seeds)} seeds) ===")
    print(f"{'='*60}")
    for r in all_results:
        print(f"    seed_{r['seed']}: {metric_name} = {r['metric_value']:.4f}")
    print(f"    Mean: {mean_metric:.4f} ± {std_metric:.4f}")
    print(f"{'='*60}\n")

    return {
        "dataset": dataset_name,
        "metric_name": metric_name,
        "per_seed": all_results,
        "mean": float(mean_metric),
        "std": float(std_metric),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune OLMo (QLoRA) on MoleculeNet with multi-seed evaluation."
    )
    parser.add_argument(
        "--datasets", nargs="+", required=True,
        help="Dataset names from experiments.yaml, or 'all' to run everything.",
    )
    parser.add_argument(
        "--config", type=str, default="experiments.yaml",
        help="Path to the experiments YAML config file.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Root directory for saving adapters and results.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print config and exit without training.",
    )
    parser.add_argument(
        "--strategy", type=str, default=None,
        help="Training strategy: 'ddp' for multi-GPU, or None for single GPU. Overrides YAML.",
    )
    args = parser.parse_args()

    # --- Load config ---
    config = load_config(args.config)
    output_root = Path(args.output_dir)

    # --- Resolve dataset list ---
    if args.datasets == ["all"]:
        dataset_names = list(config.get("datasets", {}).keys())
    else:
        dataset_names = args.datasets

    print(f"\nDatasets to run: {dataset_names}")
    print(f"Output directory: {output_root.resolve()}")

    # --- Apply CLI overrides to config ---
    if args.strategy is not None:
        config.setdefault("defaults", {})["strategy"] = args.strategy

    # --- Dry run: print config and exit ---
    if args.dry_run:
        print("\n--- DRY RUN ---")
        for name in dataset_names:
            ds_config = get_dataset_config(config, name)
            print(f"\n  [{name}]")
            for k, v in ds_config.items():
                print(f"    {k}: {v}")
        print("\nDry run complete. No training performed.")
        return

    # --- Run experiments ---
    all_dataset_results = []
    for name in dataset_names:
        result = run_dataset(config, name, output_root)
        all_dataset_results.append(result)

        # Save per-dataset results to JSON
        result_file = output_root / name / "results.json"
        result_file.parent.mkdir(parents=True, exist_ok=True)
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Results saved to: {result_file}")

    # --- Final summary ---
    print(f"\n{'#'*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'#'*70}")
    for r in all_dataset_results:
        print(f"  {r['dataset']:30s} {r['metric_name']}: {r['mean']:.4f} ± {r['std']:.4f}")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
