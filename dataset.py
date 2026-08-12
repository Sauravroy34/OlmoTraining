import torch 
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem

class MolnetDataset(Dataset):
    def __init__(self, dataset, mode="train", max_length=128,model= "harindhar10/OLMo-7B-fsdp-Pubchem-5M-1epochs",do_random = False,prob = 0.2): 
        self.mode = mode.lower()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=True,
        )
        self.tokenizer.padding_side = "right"         
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.do_random = do_random
        self.prob = prob
        tasks, datasets, transformers = dataset(featurizer="raw", splitter='scaffold')
        train, valid, test = datasets

        if self.mode == "train":
            self.data = train
        elif self.mode == "valid":
            self.data = valid 
        elif self.mode == "test":
            self.data = test
        else:
            self.data = train
        self.num_tasks = len(self.data.y[0])
    
        self.max_length = max_length
        self.samples = []
        self._filldataset()

    def _augment_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return None
        atom_order = list(range(mol.GetNumAtoms()))
        np.random.shuffle(atom_order)

        new_mol = Chem.RenumberAtoms(mol, atom_order)
        new_smiles = Chem.MolToSmiles(new_mol, canonical=False)
        del mol, new_mol
        return new_smiles

    def _filldataset(self):
        for i in range(len(self.data)):
            smiles = self.data.ids[i]
            labels = self.data.y[i]
            weights = self.data.w[i]

            # If the number of tasks is 1, filter by weight
            # If more than 1, store as multi-label and mask the loss function
            if self.num_tasks == 1:
                for task_idx, label in enumerate(labels):
                    if weights[task_idx] > 0:
                        self.samples.append((smiles, float(label)))
            else:
                self.samples.append((smiles, labels.tolist()))

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        smiles, label = self.samples[idx]
        
        if np.random.rand() < self.prob and self.do_random:
            new_smiles = self._augment_smiles(smiles)
            smiles = new_smiles if new_smiles is not None else smiles

        text = f"<|start_of_smiles|>{smiles}<|end_of_smiles|>"

        encodings = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        ) 
        
        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float),
        }
