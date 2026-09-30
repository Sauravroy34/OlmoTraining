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
import re
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
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.validation_step_outputs = []
        self.test_step_outputs = []



    def _load_model(self,checkpoint_path):

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

            #this part is for testing the
            if checkpoint_path is not None:
                peft_model = PeftModel.from_pretrained(model, checkpoint_path)
                return peft_model

            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, self.peft_config)
            return model

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            quantization_config=self.bnb_config, 
            trust_remote_code=True,
            dtype=torch.bfloat16
        )

        #this part is for testing the
        if checkpoint_path is not None:
            peft_model = PeftModel.from_pretrained(model, checkpoint_path)
            return peft_model

        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, self.peft_config)
        return model
        
    def configure_model(self,checkpoint_path = None):
        #dont call configure model if model is already
        if getattr(self, "model", None) is not None:
            return
        self.model = self._load_model(checkpoint_path)

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs.logits
        
    def get_yes_no_probs(self, logits, labels):
        yes_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        no_id = self.tokenizer.encode("No", add_special_tokens=False)[0]
        
        batch_size = labels.size(0)
        preds = []
        for i in range(batch_size):
            non_masked_idx = (labels[i] != -100).nonzero(as_tuple=True)[0]
            if len(non_masked_idx) > 0:
                first_ans_idx = non_masked_idx[0].item()
                pred_idx = first_ans_idx - 1
                
                logits_at_pred = logits[i, pred_idx]
                p_yes = torch.exp(logits_at_pred[yes_id])
                p_no = torch.exp(logits_at_pred[no_id])
                prob_yes = p_yes / (p_yes + p_no + 1e-10)
                preds.append(prob_yes.item())
            else:
                preds.append(0.5)
        return torch.tensor(preds, device=logits.device)

    def generate_and_parse_numbers(self, batch):
        batch_size = batch["input_ids"].size(0)
        preds = []
        
        # Determine max prompt length in this batch to avoid padding issues in generation
        # Better: generate per item or pad correctly on the left (but our padding is on the right)
        # We will do a loop for simplicity and correctness
        for i in range(batch_size):
            non_masked_idx = (batch["labels"][i] != -100).nonzero(as_tuple=True)[0]
            if len(non_masked_idx) > 0:
                prompt_len = non_masked_idx[0].item()
                prompt_ids = batch["input_ids"][i, :prompt_len].unsqueeze(0)
                prompt_mask = batch["attention_mask"][i, :prompt_len].unsqueeze(0)
            else:
                prompt_ids = batch["input_ids"][i].unsqueeze(0)
                prompt_mask = batch["attention_mask"][i].unsqueeze(0)
                
            generated_ids = self.model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=10,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=False
            )
            gen_text = self.tokenizer.decode(generated_ids[0, prompt_ids.size(1):], skip_special_tokens=True)
            match = re.search(r"[-+]?\d*\.\d+|\d+", gen_text)
            if match:
                preds.append(float(match.group()))
            else:
                preds.append(0.0)
                
        return torch.tensor(preds, device=batch["input_ids"].device)

    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch["labels"][..., 1:].contiguous()
        loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch["labels"][..., 1:].contiguous()
        loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        if self.task_type == "classification":
            preds = self.get_yes_no_probs(logits, batch["labels"])
        else:
            preds = self.generate_and_parse_numbers(batch)
        
        self.validation_step_outputs.append({
            "val_loss": loss.detach(),
            "preds": preds.detach() if isinstance(preds, torch.Tensor) else torch.tensor(preds, device=loss.device),
            "labels": batch["raw_labels"].detach()
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
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch["labels"][..., 1:].contiguous()
        loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if self.task_type == "classification":
            preds = self.get_yes_no_probs(logits, batch["labels"])
        else:
            preds = self.generate_and_parse_numbers(batch)

        self.test_step_outputs.append({
            "test_loss": loss.detach(),
            "preds": preds.detach() if isinstance(preds, torch.Tensor) else torch.tensor(preds, device=loss.device),
            "labels": batch["raw_labels"].detach()
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