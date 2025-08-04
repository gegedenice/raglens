#!/usr/bin/env python3
"""
Example script demonstrating the enhanced context-aware AI explanation features.
"""

import torch
import os
import sys
import time
from raglens.embeddings import EmbeddingModel
from raglens.visualization import (
    plot_token_geometry, plot_chunk_geometry, compare_pooling_methods,
    semantic_similarity_matrix, embedding_distribution_stats, layerwise_token_drift,
    chunking_length
)

def ask_user(prompt):
    ans = input(f"{prompt} [Y/n]: ").strip().lower()
    return ans in ("", "y", "yes")

def spinner(msg="Loading model..."):
    for _ in range(30):  # ~3 seconds
        for c in "|/-\\":
            sys.stdout.write(f"\r{msg} {c}")
            sys.stdout.flush()
            time.sleep(0.1)
    sys.stdout.write("\r" + " " * (len(msg) + 2) + "\r")  # Clear line

def main():
    # Ask user for preferred language and API key
    language = input("Enter your preferred language for explanations (e.g., English, French, Spanish): ").strip() or "English"
    api_key = input("Enter your OpenAI API key: ").strip()

    # Spinner while loading model
    spinner("Loading embedding model")
    model = EmbeddingModel()
    print("Embedding model loaded!\n")

    text = "The authors show that it is possible to train a transformer from scratch to perform in-context learning of linear functions."
    token_embeddings, tokens, _ = model.embed_text(text)
    chunks = [
        "Transformers can be trained from scratch for various tasks.",
        "In-context learning allows models to generalize on the fly.",
        "Linear functions are often studied in mathematics."
    ]
    query = "How can transformers learn linear functions?"

    print("=== Enhanced AI Explanation Examples ===\n")

    # 1. Token Geometry (PCA)
    if ask_user("Process Token Geometry (PCA)?"):
        try:
            fig, explanation = plot_token_geometry(
                token_embeddings, tokens, method="pca",
                generate_explanation=True, api_key=api_key, language=language
            )
            print("✅ Token Geometry with AI explanation!")
            print("🤖 AI Explanation:")
            print(explanation)
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n" + "="*60 + "\n")

    # 2. Token Geometry (UMAP)
    if ask_user("Process Token Geometry (UMAP)?"):
        try:
            fig, explanation = plot_token_geometry(
                token_embeddings, tokens, method="umap",
                generate_explanation=True, api_key=api_key, language=language
            )
            print("✅ Token Geometry (UMAP) with AI explanation!")
            print("🤖 AI Explanation:")
            print(explanation)
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n" + "="*60 + "\n")

    # 3. Pooling Comparison
    if ask_user("Process Pooling Comparison?"):
        try:
            fig, explanation = compare_pooling_methods(
                model, text, generate_explanation=True, api_key=api_key, language=language
            )
            print("✅ Pooling comparison with AI explanation!")
            print("🤖 AI Explanation:")
            print(explanation)
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n" + "="*60 + "\n")

    # 4. Chunk Geometry with Query
    if ask_user("Process Chunk Geometry with Query?"):
        try:
            fig, explanation = plot_chunk_geometry(
                chunks, model, strategy="mean", method="pca", query=query,
                generate_explanation=True, api_key=api_key, language=language
            )
            print("✅ Chunk geometry with AI explanation!")
            print("🤖 AI Explanation:")
            print(explanation)
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n" + "="*60 + "\n")

    # 5. Semantic Similarity
    if ask_user("Process Semantic Similarity Analysis?"):
        try:
            fig, explanation = semantic_similarity_matrix(
                chunks, model, generate_explanation=True, api_key=api_key, language=language
            )
            print("✅ Semantic similarity with AI explanation!")
            print("🤖 AI Explanation:")
            print(explanation)
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n" + "="*60 + "\n")

    # 6. Embedding Statistics
    if ask_user("Process Embedding Distribution Statistics?"):
        try:
            fig, explanation = embedding_distribution_stats(
                token_embeddings, generate_explanation=True, api_key=api_key, language=language
            )
            print("✅ Embedding stats with AI explanation!")
            print("🤖 AI Explanation:")
            print(explanation)
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n" + "="*60 + "\n")

    # 7. Layer-wise Token Drift
    if ask_user("Process Layer-wise Token Drift Analysis?"):
        try:
            fig, explanation = layerwise_token_drift(
                text, model, token_strs=["authors", "learning", "functions"],
                generate_explanation=True, api_key=api_key, language=language
            )
            print("✅ Token drift with AI explanation!")
            print("🤖 AI Explanation:")
            print(explanation)
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n" + "="*60 + "\n")

    # 8. Chunking Length
    if ask_user("Process Chunk Length Distribution Analysis?"):
        try:
            fig, explanation = chunking_length(
                chunks, model.tokenizer, generate_explanation=True, api_key=api_key, language=language
            )
            print("✅ Chunking length with AI explanation!")
            print("🤖 AI Explanation:")
            print(explanation)
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n" + "="*60 + "\n")

    print("✅ All examples completed!")
    print("\nAPI Key Options:")
    print("1. Environment variable: os.environ['OPENAI_API_KEY'] = 'your-key'")
    print("2. Function parameter: plot_token_geometry(..., api_key='your-key')")
    print("3. Both: Function parameter takes precedence over environment variable")
    print("\nUsage Examples:")
    print("• Normal: plot_token_geometry(embeddings, tokens)")
    print("• With AI (env var): plot_token_geometry(embeddings, tokens, generate_explanation=True)")
    print("• With AI (param): plot_token_geometry(embeddings, tokens, generate_explanation=True, api_key='key')")
    print("\nAll Functions Support AI Explanation:")
    print("• plot_token_geometry()")
    print("• plot_chunk_geometry()")
    print("• compare_pooling_methods()")
    print("• semantic_similarity_matrix()")
    print("• embedding_distribution_stats()")
    print("• layerwise_token_drift()")
    print("• chunking_length()")

if __name__ == "__main__":
    main()