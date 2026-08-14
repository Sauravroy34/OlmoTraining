%%writefile /kaggle/working/OlmoTraining/run_experiments.py
"""
Fine-tune OLMo (QLoRA) on MoleculeNet datasets with multi-seed evaluation.
"""

import argparse
import gc
import importlib
import json
import os
import shutil
import yaml
import numpy as np
import torch
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, Callback
from torch.utils.data import DataLoader

import Model
import dataset

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# ===========================================================================
# CUSTOM CALLBACK: Save ONLY LoRA Adapters (Solves Disk & RAM OOM)
# ===========================================================================
class AdapterOnlyCheckpoint(Callback):
    """
    Monitors a validation metric and saves ONLY the LoRA adapters and classifier
    head when the metric improves. Overwrites the previous best to prevent disk bloat.
    """
    def __init__(self, monitor: str, mode: str, adapter_dir: str, classifier_path: str):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.adapter_dir = adapter_dir
        self.classifier_path = classifier_path
        
        # Set initial best score depending on if we are minimizing or maximizing
        self.best_score = float("inf") if mode == "min" else float("-inf")

    def on_validation_epoch_end(self, trainer, pl_module):
        # Get the current metric
        metrics = trainer.callback_metrics
        if self.monitor not in metrics:
            return
        
        current_score = metrics[self.monitor].item()
        
        # Check if the score improved
        improved = False
        if self.mode == "min" and current_score < self.best_score:
            improved = True
        elif self.mode == "max" and current_score > self.best_score:
            improved = True

        if improved:
            self.best_score = current_score
            # Only save on the main process to avoid DDP race conditions
            if trainer.is_global_zero:
                print(f"\n  [Checkpoint] {self.monitor} improved to {self.best_score:.4f}. Saving adapters...")
                
                # Creates/Overwrites the adapter files (no cumulative bloat!)
                pl_module.model.save_pretrained(self.adapter_dir)
                
                # Overwrites the classifier head
                torch.save(pl_module.classifier.state_dict(), self.classifier_path)

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_dataset_config(config: dict, dataset_name: str) -> dict:
    defaults = config.get("defaults", {})
    datasets_cfg = config.get("datasets", {})

    if dataset_name not in datasets_cfg:
        available = ", ".join(datasets_cfg.keys())
        raise ValueError(
            f"Dataset '{dataset_name}' not found in config. Available: {available}"
        )

    ds_cfg = datasets_cfg[dataset_name]
    return {**defaults, **ds_cfg}

# ---------------------------------------------------------------------------
# DeepChem loader resolver
# ---------------------------------------------------------------------------
def resolve_loader(loader_name: str):
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
    # --- Seed everything ---
    pl.seed_everything(seed, workers=True)

    loader_fn = resolve_loader(ds_config["loader"])

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

    # Limit workers for Kaggle memory stability
    num_workers = min(2, os.cpu_count() or 2)
    
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

    # --- Paths ---
    seed_dir = output_root / dataset_name / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = str(seed_dir / "adapters")
    classifier_path = str(seed_dir / "classifier.pt")

    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name} | Seed: {seed}")
    print(f"{'='*60}")

    train_ds = dataset.MolnetDataset(
        loader_fn, mode="train", model=model_name, do_random = False
    )
    val_ds = dataset.MolnetDataset(
        loader_fn, mode="valid", model=model_name, do_random=False
    )
    test_ds = dataset.MolnetDataset(
        loader_fn, mode="test", model=model_name, do_random=False
    )

    num_tasks = train_ds.num_tasks

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
    
    # Use our custom callback instead of PyTorch Lightning's ModelCheckpoint
    adapter_checkpoint = AdapterOnlyCheckpoint(
        monitor=metric,
        mode=metric_mode,
        adapter_dir=adapter_dir,
        classifier_path=classifier_path
    )

    # --- Trainer ---
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        precision=precision,
        accelerator="gpu",
        strategy=pl_strategy,
        devices=devices,
        callbacks=[early_stopping, adapter_checkpoint],
        enable_checkpointing=False, # CRITICAL: Disables PL's 8GB .ckpt files
        enable_progress_bar=True,
        log_every_n_steps=10,
        deterministic=False,
    )

    # --- Train ---
    print(f"\n  Starting training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"  Training complete.")

    if trainer.is_global_zero:
        # We NO LONGER need to extract from a .ckpt file!
        # The custom callback has already saved the best adapters to disk.

        # --- Clean up training variables to free RAM ---
        del model
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

        # --- Evaluation: load BEST adapters onto a fresh model and test ---
        print(f"\n  Loading BEST adapters for test evaluation...")

        eval_model = Model.OlmoModelQLora(
            hparams_config=hparams_config.copy(),
            num_tasks=num_tasks,
            task_type=task_type,
            model_name=model_name,
            base_model=base_model,
        )

        eval_model.configure_model(checkpoint_path=adapter_dir)
        eval_model.classifier.load_state_dict(torch.load(classifier_path, weights_only=True))
        print(f"  Adapters and classifier loaded successfully.")

        # Cleanup classifier file from disk now that it's in memory (optional)
        if os.path.exists(classifier_path):
            os.remove(classifier_path)

        # --- Run test ---
        eval_trainer = pl.Trainer(
            precision=precision,
            accelerator="gpu",
            strategy="auto",
            devices=1,
            enable_progress_bar=True,
        )

        test_results = eval_trainer.test(eval_model, dataloaders=test_loader)

        test_metric_name = metric.replace("val_", "test_")
        test_metric_value = test_results[0].get(test_metric_name, None)
        print(f"  Test metric ({test_metric_name}): {test_metric_value}")

        # --- Final RAM & GPU memory cleanup ---
        del eval_model
        del eval_trainer
        gc.collect()
        torch.cuda.empty_cache()

        seed_result = {
            "seed": seed,
            "metric_name": test_metric_name,
            "metric_value": test_metric_value,
            "adapter_dir": adapter_dir,
        }

    else:
        seed_result = None

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    return seed_result

# ---------------------------------------------------------------------------
# Run all seeds for a dataset
# ---------------------------------------------------------------------------
def run_dataset(config: dict, dataset_name: str, output_root: Path) -> dict:
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
    parser = argparse.ArgumentParser(description="Fine-tune OLMo (QLoRA) on MoleculeNet.")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--config", type=str, default="experiments.yaml")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strategy", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_root = Path(args.output_dir)

    if args.datasets == ["all"]:
        dataset_names = list(config.get("datasets", {}).keys())
    else:
        dataset_names = args.datasets

    if args.strategy is not None:
        config.setdefault("defaults", {})["strategy"] = args.strategy

    if args.dry_run:
        print("\n--- DRY RUN ---")
        return

    all_dataset_results = []
    for name in dataset_names:
        result = run_dataset(config, name, output_root)
        all_dataset_results.append(result)

        result_file = output_root / name / "results.json"
        result_file.parent.mkdir(parents=True, exist_ok=True)
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)

    print(f"\n{'#'*70}")
    print(f"  FINAL SUMMARY")
    for r in all_dataset_results:
        print(f"  {r['dataset']:30s} {r['metric_name']}: {r['mean']:.4f} ± {r['std']:.4f}")
    print(f"{'#'*70}\n")

if __name__ == "__main__":
    main()