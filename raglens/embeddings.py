from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
from pathlib import Path

class EmbeddingModel:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", model_dir=None):
        if model_dir:
            model_dir = Path(model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / model_name.replace("/", "_")

            if not model_path.exists():
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
                tokenizer.save_pretrained(model_path)
                model.save_pretrained(model_path)
            else:
                print(f"Loading from local path: {model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path, output_hidden_states=True)

        elif Path(model_name).exists():
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

        # Model config for architecture diagnostics
        self.config = AutoConfig.from_pretrained(model_name if not Path(model_name).exists() else str(model_name))

        # Print diagnostic message at startup
        if self.supports_cls():
            print(f"Model '{model_name}' supports CLS token pooling.")
        else:
            print(f"Model '{model_name}' does NOT support CLS token pooling. Use mean/max pooling strategies.")

    def supports_cls(self):
        # Simple heuristic: check if model has pooler or typical bert config
        if hasattr(self.model, "pooler") and self.model.pooler is not None:
            return True
        # Alternatively, check model type or config
        if hasattr(self.config, "model_type"):
            if "bert" in self.config.model_type or "roberta" in self.config.model_type:
                return True
        return False

    def get_supported_strategies(self):
        strategies = ["mean", "max"]
        if self.supports_cls():
            strategies.insert(0, "cls")
        return strategies

    def embed_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", return_attention_mask=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.squeeze(0), self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0)), outputs.pooler_output

    def pool_embeddings(self, token_embeddings, attention_mask, strategy="mean"):
        if strategy == "mean":
            mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
            return (token_embeddings * mask).sum(1) / mask.sum(1)
        elif strategy == "cls":
            if not self.supports_cls():
                raise ValueError("CLS pooling not supported by this model.")
            return token_embeddings[:, 0]
        elif strategy == "max":
            mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
            token_embeddings[mask == 0] = -1e9
            return token_embeddings.max(dim=1).values
        else:
            raise ValueError("Unknown pooling strategy")

    def pooled_embedding_vector(self, token_embeddings, attention_mask, strategy="mean"):
        pooled = self.pool_embeddings(token_embeddings, attention_mask, strategy)
        return pooled.view(-1, pooled.shape[-1])

    def get_hidden_states(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", return_attention_mask=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.hidden_states, self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0)), inputs