import os
import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from typing import List

class LanguageEncoder(pl.LightningModule):
    def __init__(self, model_name, cache_dir=os.environ.get("HF_HOME", None)):
        super(LanguageEncoder, self).__init__()
        
        self.model_name = model_name
        self.cache_dir = cache_dir
        
        self.tokenizer = None
        self.language_model = None
        self._device = None
        
    def load_model(self):
        try: 
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir, padding_side="left")
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token = self.tokenizer.eos_token
        except:
            raise ValueError(f"Language model {self.model_name} not found on HuggingFace model hub. Separate implementation required.")
                        
        try:
            self.language_model = AutoModel.from_pretrained(self.model_name, cache_dir=self.cache_dir, trust_remote_code=True)
        except ValueError as e:
            print(f"Error loading model with AutoModel: {e}", "Attempting to load with AutoModelForCausalLM...", sep="\n")
            self.language_model = AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir=self.cache_dir, trust_remote_code=True)
            
        for param in self.language_model.parameters():
            param.requires_grad = False
        self.language_model.eval()
        self.language_model.to(self._device)
        
    def to(self, device):
        self._device = device
        if self.language_model is not None:
            self.language_model.to(device)
        return self
            
    def unload_model(self):
        self.tokenizer = None
        self.language_model = None
        torch.cuda.empty_cache()
    
    def forward(self, texts: List[str]):
        texts = [texts] if isinstance(texts, str) else texts
        
        if self.language_model is None:
            self.load_model()
        
        mini_batch_size = 32
        embedding_idx = 0
        embeddings = []
        successful_count = 0
        
        # Encode text in chunks to avoid running out of memory
        while True:
            try:
                with torch.no_grad():
                    input = texts[embedding_idx:min(embedding_idx+mini_batch_size, len(texts))]
                    tokens = self.tokenizer(input, return_tensors="pt", padding=True, truncation=True).to(self._device)
                    output = self.language_model(**tokens, return_dict=True)#, output_hidden_states=True)
                    embeddings.append(output.last_hidden_state[: , -1].float().squeeze())
                    
                    embedding_idx += mini_batch_size
                    successful_count += 1
            except torch.cuda.OutOfMemoryError:
                print(f"Out of memory with batch size {mini_batch_size}, encoding in smaller chunks of size {mini_batch_size // 2}")
                mini_batch_size = mini_batch_size // 2
                successful_count = 0
                if mini_batch_size < 1:
                    if len(texts[embedding_idx]) > 250:
                        mini_batch_size = 1
                        texts[embedding_idx] = texts[embedding_idx][:int(len(texts[embedding_idx]) * 0.75)]
                    else:
                        raise ValueError("Insufficient memory to encode text")
                    
            if embedding_idx >= len(texts):
                break
            if successful_count > 5:
                successful_count = 0
                mini_batch_size = int(mini_batch_size * 1.25)
                
        return torch.cat(embeddings, dim=0)