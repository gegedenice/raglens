from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    
def test_token_geometry(token_embeddings, tokens, method="pca", generate_explanation=False, api_key=None, language="English"):
    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "umap":
        if not HAS_UMAP:
            raise ImportError("UMAP is not installed. Please install it with `pip install umap-learn`.")
        reducer = UMAP(n_components=2)
    else:
        raise ValueError("Unsupported dimensionality reduction method.")

    reduced = reducer.fit_transform(token_embeddings.numpy())

    fig = plt.figure(figsize=(10, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1])
    for i, token in enumerate(tokens):
        plt.annotate(token, (reduced[i, 0], reduced[i, 1]))
    plt.title(f"Token Embeddings ({method.upper()})")
    plt.grid(True)
    # Always show the plot
    plt.show()    
    # Close the figure to prevent it from being shown again
    plt.close(fig)
    
    if generate_explanation:
        from raglens.explainer import get_base64_encoded_plot, generate_plot_explanation
        img_base64 = get_base64_encoded_plot(fig)
        explanation = generate_plot_explanation(
            img_base64, format="base64", provider="openai", api_key=api_key,
            plot_type="token_geometry", 
            text_data=tokens,
            additional_context=f"Using {method.upper()} dimensionality reduction",
            language=language
        )
        return fig, explanation
    
    return fig


def plot_token_geometry(token_embeddings, tokens, method="pca", generate_explanation=False, api_key=None, language="English"):
    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "umap":
        if not HAS_UMAP:
            raise ImportError("UMAP is not installed. Please install it with `pip install umap-learn`.")
        reducer = UMAP(n_components=2)
    else:
        raise ValueError("Unsupported dimensionality reduction method.")

    reduced = reducer.fit_transform(token_embeddings.numpy())

    fig = plt.figure(figsize=(10, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1])
    for i, token in enumerate(tokens):
        plt.annotate(token, (reduced[i, 0], reduced[i, 1]))
    plt.title(f"Token Embeddings ({method.upper()})")
    plt.grid(True)
    # Always show the plot
    plt.show()    
    # Close the figure to prevent it from being shown again
    plt.close(fig)
    
    if generate_explanation:
        from raglens.explainer import get_base64_encoded_plot, generate_plot_explanation
        img_base64 = get_base64_encoded_plot(fig)
        explanation = generate_plot_explanation(
            img_base64, format="base64", provider="openai", api_key=api_key,
            plot_type="token_geometry", 
            text_data=tokens,
            additional_context=f"Using {method.upper()} dimensionality reduction",
            language=language
        )
        return fig, explanation
    
    return fig
    
def plot_chunk_geometry(chunks, embedding_model, strategy="mean", method="pca", query=None, generate_explanation=False, api_key=None, language="English"):
    chunk_embs = []
    for chunk in chunks:
        inputs = embedding_model.tokenizer(chunk, return_tensors="pt", return_attention_mask=True)
        with torch.no_grad():
            outputs = embedding_model.model(**inputs)
        pooled = embedding_model.pooled_embedding_vector(outputs.last_hidden_state, inputs["attention_mask"], strategy)
        chunk_embs.append(pooled.squeeze(0))

    chunk_embs = torch.stack(chunk_embs)

    #labels = [f"chunk_{i}" for i in range(len(chunks))]
    labels = [f"{chunk[:25]}{'...' if len(chunk) > 25 else ''}" for chunk in chunks]

    if query:
        q_inputs = embedding_model.tokenizer(query, return_tensors="pt", return_attention_mask=True)
        with torch.no_grad():
            q_outputs = embedding_model.model(**q_inputs)
        q_pooled = embedding_model.pooled_embedding_vector(q_outputs.last_hidden_state, q_inputs["attention_mask"], strategy)
        chunk_embs = torch.cat([chunk_embs, q_pooled], dim=0)
        labels.append("query")

    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "umap":
        if not HAS_UMAP:
            raise ImportError("UMAP is not installed. Please install it with `pip install umap-learn`.")
        reducer = UMAP(n_components=2)
    else:
        raise ValueError("Unsupported dimensionality reduction method.")

    reduced = reducer.fit_transform(chunk_embs.numpy())

    fig = plt.figure(figsize=(10, 6))
    for i, label in enumerate(labels):
        color = "red" if label == "query" else "blue"
        plt.scatter(reduced[i, 0], reduced[i, 1], color=color)
        plt.annotate(label, (reduced[i, 0], reduced[i, 1]))

    plt.title(f"Chunk Embeddings ({method.upper()}, {strategy.upper()} pooling)")
    plt.grid(True)   
    # Always show the plot
    plt.show()    
    # Close the figure to prevent it from being shown again
    plt.close(fig)
    
    if generate_explanation:
        from raglens.explainer import get_base64_encoded_plot, generate_plot_explanation
        img_base64 = get_base64_encoded_plot(fig)
        
        additional_context = f"Using {strategy.upper()} pooling and {method.upper()} dimensionality reduction"
        if query:
            additional_context += f". Query: '{query}'"
        
        explanation = generate_plot_explanation(
            img_base64, format="base64", provider="openai", api_key=api_key,
            plot_type="chunk_geometry", 
            text_data=chunks,
            additional_context=additional_context,
            language=language
        )
        return fig, explanation
    
    return fig
    
def compare_pooling_methods(embedding_model, text, strategies=None, generate_explanation=False, api_key=None, language="English"):
    if strategies is None:
        strategies = embedding_model.get_supported_strategies()
    inputs = embedding_model.tokenizer(text, return_tensors="pt", return_attention_mask=True)
    with torch.no_grad():
        outputs = embedding_model.model(**inputs)

    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"]

    pooled = {}
    for strategy in strategies:
        pooled_output = embedding_model.pooled_embedding_vector(token_embeddings, attention_mask, strategy)
        pooled[strategy] = pooled_output

    similarities = torch.zeros((len(strategies), len(strategies)))
    for i, s1 in enumerate(strategies):
        for j, s2 in enumerate(strategies):
            sim = F.cosine_similarity(pooled[s1], pooled[s2]).item()
            similarities[i, j] = sim

    fig = plt.figure(figsize=(6, 5))
    plt.imshow(similarities.numpy(), cmap="viridis", interpolation="nearest")
    plt.colorbar()
    plt.xticks(range(len(strategies)), strategies)
    plt.yticks(range(len(strategies)), strategies)
    plt.title("Cosine Similarity between Pooling Methods")
    # Always show the plot
    plt.show()    
    # Close the figure to prevent it from being shown again
    plt.close(fig)
    
    if generate_explanation:
        from raglens.explainer import get_base64_encoded_plot, generate_plot_explanation
        img_base64 = get_base64_encoded_plot(fig)
        explanation = generate_plot_explanation(
            img_base64, format="base64", provider="openai", api_key=api_key,
            plot_type="pooling_comparison", 
            text_data=text,
            additional_context=f"Comparing strategies: {strategies}",
            language=language
        )
        return fig, explanation
    
    return fig
    
def semantic_similarity_matrix(chunks, embedding_model, strategies=None, generate_explanation=False, api_key=None, language="English"):
    if strategies is None:
        strategies = embedding_model.get_supported_strategies()
    fig, axes = plt.subplots(1, len(strategies), figsize=(6 * len(strategies), 6))
    if len(strategies) == 1:
        axes = [axes]

    for idx, strategy in enumerate(strategies):
        embs = []
        for chunk in chunks:
            token_embs, _, _ = embedding_model.embed_text(chunk)
            inputs = embedding_model.tokenizer(chunk, return_tensors="pt", return_attention_mask=True)
            pooled = embedding_model.pooled_embedding_vector(token_embs.unsqueeze(0), inputs["attention_mask"], strategy=strategy)
            embs.append(pooled.squeeze(0))

        embs = torch.stack(embs)
        sims = torch.matmul(embs, embs.T)
        norms = torch.norm(embs, dim=1, keepdim=True)
        sims = sims / (norms @ norms.T)

        ax = axes[idx]
        ax.imshow(sims.numpy(), cmap="coolwarm", vmin=0, vmax=1)
        ax.set_title(f"Cosine Similarity ({strategy.upper()})", fontsize=14)
        ax.set_xticks(range(len(chunks)))
        ax.set_xticklabels([f"S{i+1}" for i in range(len(chunks))], rotation=45)
        ax.set_yticks(range(len(chunks)))
        ax.set_yticklabels([f"S{i+1}" for i in range(len(chunks))])
           
    plt.tight_layout()
    # Always show the plot
    plt.show()    
    # Close the figure to prevent it from being shown again
    plt.close(fig)
    
    if generate_explanation:
        from raglens.explainer import get_base64_encoded_plot, generate_plot_explanation
        img_base64 = get_base64_encoded_plot(fig)
        explanation = generate_plot_explanation(
            img_base64, format="base64", provider="openai", api_key=api_key,
            plot_type="semantic_similarity", 
            text_data=chunks,
            additional_context=f"Using strategies: {strategies}",
            language=language
        )
        return fig, explanation
    
    return fig
    
def chunking_length(chunks, tokenizer, generate_explanation=False, api_key=None, language="English"):
    lengths = [len(tokenizer(chunk)["input_ids"]) for chunk in chunks]
    
    fig = plt.figure(figsize=(8, 4))
    plt.bar(range(len(chunks)), lengths)
    plt.xlabel("Chunk Index")
    plt.ylabel("Token Count")
    plt.title("Chunk Length Distribution")
    plt.grid(True)
    # Always show the plot
    plt.show()    
    # Close the figure to prevent it from being shown again
    plt.close(fig)
    
    if generate_explanation:
        from raglens.explainer import get_base64_encoded_plot, generate_plot_explanation
        img_base64 = get_base64_encoded_plot(fig)
        explanation = generate_plot_explanation(
            img_base64, format="base64", provider="openai", api_key=api_key,
            plot_type="chunking_length", 
            text_data=chunks,
            additional_context=f"Number of chunks: {len(chunks)}, Mean length: {np.mean(lengths):.1f} tokens, Min: {min(lengths)}, Max: {max(lengths)}",
            language=language
        )
        return fig, explanation
    
    return fig
    
def chunking_sanity(chunks, tokenizer, highlight_overlap=True):
    print("\n🧩 Chunking Diagnostics\n")
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} ---")
        tokens = tokenizer.tokenize(chunk)
        print(f"Tokens: {tokens[:15]}{'...' if len(tokens) > 15 else ''}")
        ends_with_punct = chunk.strip()[-1] in {".", "!", "?"}
        print("Ends cleanly:", "✅" if ends_with_punct else "⚠️ Possibly mid-sentence")
        print(f"Token count: {len(tokens)}")
        print("")

    if highlight_overlap:
        print("🔁 Overlap Check")
        for i in range(1, len(chunks)):
            overlap = set(chunks[i-1].split()) & set(chunks[i].split())
            if overlap:
                print(f"Chunk {i} overlaps with {i-1}: {sorted(overlap)}")
        print("")

def embedding_distribution_stats(embeddings, generate_explanation=False, api_key=None, language="English"):
    norms = torch.norm(embeddings, dim=1)
    pca = PCA()
    explained = pca.fit(embeddings.numpy()).explained_variance_ratio_

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist(norms.numpy(), bins=20, color="gray")
    ax1.set_title("Embedding Norm Distribution")
    ax1.set_xlabel("L2 Norm")

    ax2.plot(np.cumsum(explained), marker="o")
    ax2.set_title("PCA Explained Variance")
    ax2.set_xlabel("Principal Component")
    ax2.set_ylabel("Cumulative Variance Ratio")
    ax2.grid(True)

    plt.tight_layout()
    # Always show the plot
    plt.show()    
    # Close the figure to prevent it from being shown again
    plt.close(fig)
    
    if generate_explanation:
        from raglens.explainer import get_base64_encoded_plot, generate_plot_explanation
        img_base64 = get_base64_encoded_plot(fig)
        explanation = generate_plot_explanation(
            img_base64, format="base64", provider="openai", api_key=api_key,
            plot_type="embedding_stats", 
            additional_context=f"Embedding shape: {embeddings.shape}, Mean norm: {norms.mean():.3f}, Explained variance (first 10 PCs): {explained[:10].sum():.3f}",
            language=language
        )
        return fig, explanation
    
    return fig
    
def layerwise_token_drift(text, embedding_model, token_strs, generate_explanation=False, api_key=None, language="English"):
    hidden_states, tokens, inputs = embedding_model.get_hidden_states(text)
    input_ids = inputs["input_ids"][0]
    token_indices = [i for i, tok in enumerate(tokens) if tok in token_strs]

    fig = plt.figure(figsize=(10, 6))
    for token_index in token_indices:
        token = tokens[token_index]
        trajectory = [layer[0, token_index, :].numpy() for layer in hidden_states]
        dists = [np.linalg.norm(trajectory[i + 1] - trajectory[i]) for i in range(len(trajectory) - 1)]
        plt.plot(dists, label=token)

    plt.title("Layer-wise Token Drift")
    plt.xlabel("Layer")
    plt.ylabel("L2 Distance to Previous Layer")
    plt.legend()
    plt.grid(True)
    # Always show the plot
    plt.show()    
    # Close the figure to prevent it from being shown again
    plt.close(fig)
    
    if generate_explanation:
        from raglens.explainer import get_base64_encoded_plot, generate_plot_explanation
        img_base64 = get_base64_encoded_plot(fig)
        explanation = generate_plot_explanation(
            img_base64, format="base64", provider="openai", api_key=api_key,
            plot_type="token_drift", 
            text_data=token_strs,
            additional_context=f"Analyzing drift for tokens: {token_strs} in text: '{text[:100]}{'...' if len(text) > 100 else ''}'",
            language=language
        )
        return fig, explanation
    
    return fig
# %%
