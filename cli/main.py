### File: cli/main.py
import argparse
import os
from raglens.embeddings import EmbeddingModel
from raglens.visualization import (
    plot_token_geometry,
    compare_pooling_methods,
    semantic_similarity_matrix,
    chunking_length,
    chunking_sanity,
    embedding_distribution_stats,
    plot_chunk_geometry,
    layerwise_token_drift
)
from raglens.retrieval import ChunkRetriever, compare_retrieval_pooling


def main():
    parser = argparse.ArgumentParser(description="Raglens CLI Toolkit")

    parser.add_argument("--mode", choices=[
        "token-geometry", "pooling-compare", "semantic-similarity",
        "chunk-sanity", "embedding-stats", "chunk-geometry",
        "token-drift", "retrieval-compare"
    ], required=True, help="Analysis mode")

    parser.add_argument("--text", type=str, help="Input text for single-pass tasks")
    parser.add_argument("--chunks", type=str, nargs="*", help="List of text chunks for chunk-based tasks")
    parser.add_argument("--query", type=str, help="Optional query for chunk-based tasks")
    parser.add_argument("--model-name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--strategy", type=str, choices=["cls", "mean", "max"], help="Single pooling strategy")
    parser.add_argument("--strategies", type=str, help="Comma-separated list of pooling strategies")
    parser.add_argument("--reduction", type=str, choices=["pca", "umap"], default="pca")
    parser.add_argument("--generate-explanation", action="store_true", help="Generate explanation for the plot")
    parser.add_argument("--api-key", type=str, help="OpenAI API key for explanation generation")
    parser.add_argument("--language", type=str, default="English", help="Language for AI explanations")

    args = parser.parse_args()

    # If api-key is provided, set it as env variable for consistency
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key

    model = EmbeddingModel(model_name=args.model_name, model_dir=args.model_dir)

    if args.strategies:
        strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    else:
        strategies = model.get_supported_strategies()

    # Helper to print explanation if present
    def handle_output(result):
        if isinstance(result, tuple):
            fig, explanation = result
            print(explanation)
        else:
            fig = result
        return fig

    if args.mode == "token-geometry":
        if not args.text:
            raise ValueError("--text is required for token-geometry mode")
        token_embs, tokens, _ = model.embed_text(args.text)
        result = plot_token_geometry(
            token_embs, tokens, method=args.reduction,
            generate_explanation=args.generate_explanation,
            api_key=args.api_key,
            language=args.language
        )
        handle_output(result)

    elif args.mode == "pooling-compare":
        if not args.text:
            raise ValueError("--text is required for pooling-compare mode")
        result = compare_pooling_methods(
            model, args.text, strategies=strategies,
            generate_explanation=args.generate_explanation,
            api_key=args.api_key,
            language=args.language
        )
        handle_output(result)

    elif args.mode == "semantic-similarity":
        if not args.chunks:
            raise ValueError("--chunks is required for semantic-similarity mode")
        result = semantic_similarity_matrix(
            args.chunks, model, strategies=strategies,
            generate_explanation=args.generate_explanation,
            api_key=args.api_key,
            language=args.language
        )
        handle_output(result)

    elif args.mode == "chunk-sanity":
        if not args.chunks:
            raise ValueError("--chunks is required for chunk-sanity mode")
        chunks = args.chunks
        result = chunking_length(
            chunks, model.tokenizer,
            generate_explanation=args.generate_explanation,
            api_key=args.api_key,
            language=args.language
        )
        handle_output(result)
        chunking_sanity(chunks, model.tokenizer)

    elif args.mode == "embedding-stats":
        if not args.text:
            raise ValueError("--text is required for embedding-stats mode")
        token_embs, _, _ = model.embed_text(args.text)
        result = embedding_distribution_stats(
            token_embs, generate_explanation=args.generate_explanation,
            api_key=args.api_key,
            language=args.language
        )
        handle_output(result)

    elif args.mode == "chunk-geometry":
        if not args.chunks:
            raise ValueError("--chunks is required for chunk-sanity mode")
        chunks = args.chunks
        result = plot_chunk_geometry(
            chunks, model, method=args.reduction, query=args.query,
            generate_explanation=args.generate_explanation,
            api_key=args.api_key,
            language=args.language
        )
        handle_output(result)

    elif args.mode == "token-drift":
        if not args.text:
            raise ValueError("--text is required for token-drift mode")
        token_strs = input("Enter token substrings to track (comma-separated): ").split(",")
        token_strs = [t.strip() for t in token_strs if t.strip()]
        result = layerwise_token_drift(
            args.text, model, token_strs,
            generate_explanation=args.generate_explanation,
            api_key=args.api_key,
            language=args.language
        )
        handle_output(result)

    elif args.mode == "retrieval-compare":
        if not args.query:
            raise ValueError("--query is required for retrieval-compare mode")
        if not args.chunks:
            raise ValueError("--chunks is required for retrieval-compare mode")
        chunks = args.chunks
        retriever = ChunkRetriever(model)
        retriever.add_chunks(chunks)
        compare_retrieval_pooling(retriever, args.query)
        result = plot_chunk_geometry(
            chunks, model, method=args.reduction, query=args.query,
            generate_explanation=args.generate_explanation,
            api_key=args.api_key,
            language=args.language
        )
        handle_output(result)


if __name__ == "__main__":
    main()
