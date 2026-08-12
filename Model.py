import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel
)
from sklearn.metrics import roc_auc_score, mean_squared_error
import numpy as np

class OlmoModelQLora(pl.LightningModule):
    def __init__(self, hparams_config, num_tasks,task_type, model_name,base_model):
        super().__init__()
        self.save_hyperparameters(hparams_config)
        self.num_tasks = num_tasks
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        self.task_type = task_type.lower() # classification or regression
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_name = model_name
        self.base_model = base_model

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.peft_config = LoraConfig(
            r=self.hparams.lora_r,
            lora_alpha=self.hparams.lora_alpha,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            lora_dropout=self.hparams.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        if self.task_type == "classification":
            self.loss_fn = self._masked_bce
        elif self.task_type == "regression":
            self.loss_fn = self._masked_mse
        else:
            raise ValueError("Task type must be classification or regression")
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def _masked_bce(self,logits,labels):
        mask = ~torch.isnan(labels)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits[mask], labels[mask], reduction="mean"
        )
        return loss

    def _masked_mse(self, preds, labels):
        mask = ~torch.isnan(labels)
        loss = nn.functional.mse_loss(
            preds[mask], labels[mask], reduction="mean"
        )
        return loss

    def _load_model(self):

        from huggingface_hub import model_info
        info = model_info(self.model_name)
        file_names = [f.rfilename for f in info.siblings]
        
        # checks if the given path is an adaptor or a model
        if "adapter_config.json" in file_names:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model, 
                quantization_config=self.bnb_config, 
                trust_remote_code=True,
                dtype=torch.bfloat16
            )
            model = PeftModel.from_pretrained(base_model, self.model_name)
            model = model.merge_and_unload()
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, self.peft_config)
            return model

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            quantization_config=self.bnb_config, 
            trust_remote_code=True,
            dtype=torch.bfloat16
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, self.peft_config)
        return model
        
    def configure_model(self):
        self.model = self._load_model()
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.gradient_checkpointing_enable()
            
        hidden_size = self.model.config.hidden_size    
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, self.num_tasks)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True
        )
        last_hidden_states = outputs.hidden_states[-1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.shape[0]
        
        final_token_hidden = last_hidden_states[torch.arange(batch_size), sequence_lengths]
        logits = self.classifier(final_token_hidden)
        return logits
        
    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(logits.squeeze(-1), batch["labels"].float())
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(logits.squeeze(-1), batch["labels"].float())
        
        if self.task_type == "classification":
            preds = torch.sigmoid(logits.squeeze(-1))
        else:
            preds = logits.squeeze(-1)
        
        # Store predictions for epoch-end calculation
        self.validation_step_outputs.append({
            "val_loss": loss.detach(),
            "preds": preds.detach(),
            "labels": batch["labels"].detach()
        })
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        all_preds = torch.cat([x["preds"] for x in self.validation_step_outputs]).float().cpu().numpy()
        all_labels = torch.cat([x["labels"] for x in self.validation_step_outputs]).float().cpu().numpy()
        self.validation_step_outputs.clear()
        
        # Mask out NaN values for metric calculations
        mask = ~np.isnan(all_labels)

        if self.task_type == "classification":
            try:
                val_metric = roc_auc_score(all_labels[mask], all_preds[mask])
            except ValueError:
                val_metric = 0.5
            
            # Log metric specifically for Optuna tracking
            self.log("val_roc_auc", val_metric, prog_bar=True, sync_dist=True)
        elif self.task_type == "regression":
            val_metric = np.sqrt(mean_squared_error(all_labels[mask], all_preds[mask]))
            
            # Log metric specifically for Optuna tracking
            self.log("val_rmse", val_metric, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(logits.squeeze(-1), batch["labels"].float())

        if self.task_type == "classification":
            preds = torch.sigmoid(logits.squeeze(-1))
        else:
            preds = logits.squeeze(-1)

        self.test_step_outputs.append({
            "test_loss": loss.detach(),
            "preds": preds.detach(),
            "labels": batch["labels"].detach()
        })
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return

        all_preds = torch.cat([x["preds"] for x in self.test_step_outputs]).float().cpu().numpy()
        all_labels = torch.cat([x["labels"] for x in self.test_step_outputs]).float().cpu().numpy()
        self.test_step_outputs.clear()

        mask = ~np.isnan(all_labels)

        if self.task_type == "classification":
            try:
                test_metric = roc_auc_score(all_labels[mask], all_preds[mask])
            except ValueError:
                test_metric = 0.5
            self.log("test_roc_auc", test_metric, prog_bar=True, sync_dist=True)
        elif self.task_type == "regression":
            test_metric = np.sqrt(mean_squared_error(all_labels[mask], all_preds[mask]))
            self.log("test_rmse", test_metric, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params, 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.15 * total_steps)
        
        scheduler_warmup = LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        )
        scheduler_cosine = CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - warmup_steps)
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler_warmup, scheduler_cosine],
            milestones=[warmup_steps]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"}
        }