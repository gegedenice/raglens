import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

class ChunkRetriever:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.strategies = self.embedding_model.get_supported_strategies()
        self.chunks = []
        self.embeddings = []

    def add_chunks(self, chunk_list, pooling="mean"):
        for chunk in chunk_list:
            inputs = self.embedding_model.tokenizer(chunk, return_tensors="pt", return_attention_mask=True)
            with torch.no_grad():
                outputs = self.embedding_model.model(**inputs)
            token_embeddings = outputs.last_hidden_state
            pooled = self.embedding_model.pool_embeddings(token_embeddings, inputs["attention_mask"], strategy=pooling)
            self.chunks.append(chunk)
            self.embeddings.append(pooled.squeeze(0))
        self.embeddings = torch.stack(self.embeddings)

    def retrieve(self, query, top_k=5, pooling="mean"):
        inputs = self.embedding_model.tokenizer(query, return_tensors="pt", return_attention_mask=True)
        with torch.no_grad():
            outputs = self.embedding_model.model(**inputs)
        token_embeddings = outputs.last_hidden_state
        query_embedding = self.embedding_model.pool_embeddings(token_embeddings, inputs["attention_mask"], strategy=pooling).squeeze(0)

        similarities = F.cosine_similarity(query_embedding.unsqueeze(0), self.embeddings)
        top_k_indices = torch.topk(similarities, top_k).indices

        return [(self.chunks[i], similarities[i].item()) for i in top_k_indices]
    
def compare_retrieval_pooling(retriever, query, top_k=3):
    results_by_strategy = {}
    strategies = retriever.strategies
    for strategy in strategies:
        results = retriever.retrieve(query, top_k=top_k, pooling=strategy)
        results_by_strategy[strategy] = results

    print(f"\n🔍 Retrieval results for query: '{query}'\n")
    for strategy in strategies:
        print(f"--- Pooling Strategy: {strategy.upper()} ---")
        for score, chunk in sorted([(s, c) for c, s in results_by_strategy[strategy]], reverse=True):
            print(f"[{score:.3f}] {chunk}")
        print("")
