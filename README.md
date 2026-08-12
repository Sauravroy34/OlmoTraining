# OLMo QLoRA Fine-Tuning for MoleculeNet

This repository contains a simple, modular framework for fine-tuning the OLMo language model using QLoRA on various MoleculeNet datasets. 

It handles training with multi-seed evaluation, early stopping, and automatic LoRA adapter saving.

## Getting Started

### Configuration

All experiment parameters are defined in `experiments.yaml`. You can set global defaults (like model name, base model, precision) and dataset-specific parameters (like learning rate, batch size, epochs).

To add a new dataset or modify hyper-parameters, simply edit the `experiments.yaml` file.

### Running Experiments

You can run experiments using the `run_experiments.py` script. Here are some common usage examples:

1. **Run a specific dataset:**
   ```bash
   python run_experiments.py --datasets bbbp
   ```

2. **Run multiple datasets:**
   ```bash
   python run_experiments.py --datasets bbbp bace_classification
   ```

3. **Run all datasets defined in the config:**
   ```bash
   python run_experiments.py --datasets all
   ```

4. **Multi-GPU Training (DDP):**
   ```bash
   python run_experiments.py --datasets bbbp --strategy ddp
   ```

5. **Dry Run (Check config without training):**
   ```bash
   python run_experiments.py --datasets all --dry-run
   ```

### Output Structure

By default, all models and results will be saved to a `results/` directory, structured as follows:

```text
results/
└── {dataset_name}/
    ├── results.json (aggregated results over seeds)
    ├── seed_42/
    │   ├── adapters/     # LoRA adapters
    │   ├── checkpoints/  # Best checkpoints
    │   └── classifier.pt # Classifier head
    ├── seed_43/
    └── seed_44/
```

### Custom Configuration

If you have multiple YAML configurations, you can pass them to the script using `--config`:
```bash
python run_experiments.py --datasets bbbp --config custom_config.yaml
```
