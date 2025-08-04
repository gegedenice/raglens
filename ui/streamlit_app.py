import streamlit as st
import os
from raglens.embeddings import EmbeddingModel
from raglens.visualization import (
    plot_token_geometry, plot_chunk_geometry, compare_pooling_methods,
    semantic_similarity_matrix, embedding_distribution_stats, layerwise_token_drift,
    chunking_length, chunking_sanity
)
from raglens.retrieval import ChunkRetriever, compare_retrieval_pooling
import io
from contextlib import redirect_stdout

st.set_page_config(page_title="Raglens Visual Diagnostics", layout="wide")

st.title("🔬 Raglens: Visual Diagnostics for Embedding Models in RAG Pipelines")
st.markdown("""
A user-friendly interface for exploring token and chunk embeddings, pooling strategies, retrieval diagnostics, and AI-powered plot explanations.
""")

# Sidebar: Model selection and API key
st.sidebar.header("Model & API Settings")
model_name = st.sidebar.text_input("Model Name", "sentence-transformers/all-MiniLM-L6-v2")
model_dir = st.sidebar.text_input("Model Directory (optional)", "")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
language = st.sidebar.selectbox("Explanation Language", ["English", "French", "Spanish", "German", "Italian", "Chinese", "Japanese", "Other"])
generate_explanation = st.sidebar.checkbox("Generate AI Explanation", value=False)

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

# Load model
@st.cache_resource(show_spinner=True)
def load_model(model_name, model_dir):
    return EmbeddingModel(model_name=model_name, model_dir=model_dir or None)

model = load_model(model_name, model_dir)
tokenizer = model.tokenizer

st.success(f"Model '{model_name}' loaded!")

text_example = "The authors show that it is possible to train a transformer from scratch to perform in-context learning of linear functions."
chunks_example = "\n".join([
                            "Transformers can be trained from scratch for various tasks.",
                            "In-context learning allows models to generalize on the fly.",
                            "Linear functions are often studied in mathematics.",
                            "The authors describe a novel approach to training models.",
                            "Tokenization is a crucial step in NLP pipelines."
                        ])
query_example = "How can transformers learn linear functions from scratch?"

# Tabs for each feature
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Token-level: Token Geometry", "Token-level: Layerwise Drift", 
    "Sentence-level: Pooling Comparison",  "Sentence-level: Embedding Stats",
    "Chunks-level: Chunking Sanity", "Chunks-level: Chunk Geometry", "Chunks-level: Semantic Similarity", 
    "Retrieval"
])

with tab1:
    st.header("Token-level: Token Geometry (PCA/UMAP)")
    st.markdown("Visualize token embeddings in 2D using PCA or UMAP. This helps you understand how tokens are distributed in the embedding space and spot semantic clusters or outliers.",
                help="Visualisez les embeddings de tokens en 2D avec PCA ou UMAP. Cela permet de comprendre la distribution des tokens dans l'espace d'embedding et d'identifier les clusters sémantiques ou les anomalies.")
    text = st.text_area(
        "Input Text",
        text_example,
        key="token_geometry_text"
    )
    method = st.selectbox("Dimensionality Reduction", ["pca", "umap"], key="token_geometry_method")
    if st.button("Visualize Token Geometry", key="token_geometry_btn"):
        token_embeddings, tokens, _ = model.embed_text(text)
        result = plot_token_geometry(
            token_embeddings, tokens, method=method,
            generate_explanation=generate_explanation, api_key=api_key, language=language
        )
        if generate_explanation and isinstance(result, tuple):
            fig, explanation = result
            st.pyplot(fig)
            st.info(explanation)
        else:
            st.pyplot(result)

with tab2:
    st.header("Token-level: Layerwise Token Drift")
    st.markdown(
        "Track how selected token representations change across transformer layers. Useful for understanding how information propagates and transforms through the model.",
        help="Suivez l'évolution des représentations de tokens à travers les couches du transformeur. Utile pour comprendre comment l'information se propage et se transforme dans le modèle."
    )
    text2 = st.text_area("Input Text for Drift", text_example, key="drift_text")
    token_strs = st.text_input("Tokens to Track (comma-separated)", "authors,learning,functions", key="drift_tokens")
    token_list = [t.strip() for t in token_strs.split(",") if t.strip()]
    if st.button("Show Layerwise Drift", key="drift_btn"):
        result = layerwise_token_drift(
            text2, model, token_list,
            generate_explanation=generate_explanation, api_key=api_key, language=language
        )
        if generate_explanation and isinstance(result, tuple):
            fig, explanation = result
            st.pyplot(fig)
            st.info(explanation)
        else:
            st.pyplot(result)
            
with tab3:
    st.header("Sentence-level: Pooling Comparison")
    st.markdown("Compare sentence-level embeddings using different pooling strategies (CLS, mean, max). Useful for selecting the best representation for downstream tasks.",
                help="Comparez les embeddings d'une phrase selon différentes stratégies de pooling (CLS, moyenne, max). Utile pour choisir la meilleure représentation pour les tâches en aval.")
    text3 = st.text_area("Input Text for Pooling", text_example, key="pooling_text")
    strategies = st.multiselect("Pooling Strategies", model.get_supported_strategies(), default=model.get_supported_strategies(), key="pooling_strategies")
    if st.button("Compare Pooling Methods", key="pooling_btn"):
        result = compare_pooling_methods(
            model, text3, strategies=strategies,
            generate_explanation=generate_explanation, api_key=api_key, language=language
        )
        if generate_explanation and isinstance(result, tuple):
            fig, explanation = result
            st.pyplot(fig)
            st.info(explanation)
        else:
            st.pyplot(result)

with tab4:
    st.header("Sentence-level: Embedding Distribution Statistics")
    st.markdown(
        "Plot distributional statistics (e.g., L2 norms, explained variance) for token embeddings. Useful for detecting anomalies or understanding embedding magnitude and variance.",
        help="Affichez les statistiques de distribution (normes L2, variance expliquée) des embeddings de tokens. Utile pour détecter des anomalies ou comprendre la magnitude et la variance des embeddings."
    )
    text4 = st.text_area("Input Text for Stats", text_example, key="stats_text")
    if st.button("Show Embedding Stats", key="stats_btn"):
        token_embeddings, _, _ = model.embed_text(text4)
        result = embedding_distribution_stats(
            token_embeddings, generate_explanation=generate_explanation, api_key=api_key, language=language
        )
        if generate_explanation and isinstance(result, tuple):
            fig, explanation = result
            st.pyplot(fig)
            st.info(explanation)
        else:
            st.pyplot(result)

with tab5:
    st.header("Chunks-level: Chunking Sanity Check")
    st.markdown(
        "Inspect how tokenization splits long texts into chunks. Highlights chunk boundaries and overlap, ensuring proper chunking for retrieval.",
        help="Inspectez comment la tokenisation découpe les textes longs en chunks. Met en évidence les frontières et les chevauchements, pour garantir un découpage adapté à la recherche."
    )
    chunk_list = st.text_area("Chunks (one per line)", chunks_example, key="chunking_chunks").splitlines()
    if st.button("Show Chunking", key="chunking_btn"):
        if not(chunk_list and any(chunk_list)):
            st.error("Please provide some chunks or text to chunk.")
        chunks = [c for c in chunk_list if c.strip()]
        fig = chunking_length(chunks, tokenizer, generate_explanation=generate_explanation, api_key=api_key, language=language)
        if generate_explanation and isinstance(fig, tuple):
            fig, explanation = fig
            st.pyplot(fig)
            st.info(explanation)
        else:
            st.pyplot(fig)
        st.write("Chunking Sanity Table:")
        #chunking_sanity(chunks, tokenizer, highlight_overlap=True)
        with io.StringIO() as buf, redirect_stdout(buf):
            chunking_sanity(chunks, tokenizer, highlight_overlap=True)
            output = buf.getvalue()
        st.text(output)

with tab6:
    st.header("Chunks-level: Chunk Geometry Visualization")
    st.markdown(
        "Scatter plot chunk embeddings (optionally with a query) in 2D space, colored by pooling strategy. Reveals chunk clustering and query proximity.",
        help="Affichez les embeddings de chunks (et éventuellement une requête) dans un espace 2D, colorés selon la stratégie de pooling. Permet de visualiser les regroupements et la proximité avec la requête."
    )
    chunk_list2 = st.text_area("Or paste chunks (one per line)", chunks_example, key="chunk_geom_chunks").splitlines()
    strategy = st.selectbox("Pooling Strategy", model.get_supported_strategies(), key="chunk_geom_strategy")
    method2 = st.selectbox("Dimensionality Reduction", ["pca", "umap"], key="chunk_geom_method")
    query = st.text_input("Optional Query", "", key="chunk_geom_query")
    if st.button("Show Chunk Geometry", key="chunk_geom_btn"):
        if not(chunk_list2 and any(chunk_list2)):
            st.error("Please provide some chunks or text to chunk.")
        chunks2 = [c for c in chunk_list2 if c.strip()]
        result = plot_chunk_geometry(
            chunks2, model, strategy=strategy, method=method2, query=query,
            generate_explanation=generate_explanation, api_key=api_key, language=language
        )
        if generate_explanation and isinstance(result, tuple):
            fig, explanation = result
            st.pyplot(fig)
            st.info(explanation)
        else:
            st.pyplot(result)

with tab7:
    st.header("Chunks-level: Semantic Similarity Matrix")
    st.markdown(
        "Display a heatmap of inter-sentence or inter-chunk similarities. Useful for visualizing semantic relationships and redundancy.",
        help="Affichez une matrice de similarité entre phrases ou chunks. Utile pour visualiser les relations sémantiques et les redondances."
    )
    chunk_list3 = st.text_area("Or paste chunks (one per line)", chunks_example, key="similarity_chunks").splitlines()
    strategies2 = st.multiselect("Pooling Strategies", model.get_supported_strategies(), default=model.get_supported_strategies(), key="similarity_strategies")
    if st.button("Show Semantic Similarity", key="similarity_btn"):
        result = semantic_similarity_matrix(
            chunk_list3, model, strategies=strategies2,
            generate_explanation=generate_explanation, api_key=api_key, language=language
        )
        if generate_explanation and isinstance(result, tuple):
            fig, explanation = result
            st.pyplot(fig)
            st.info(explanation)
        else:
            st.pyplot(result)

with tab8:
    st.header("Retrieval Diagnostics")
    st.markdown(
        "Visualize the impact of different pooling strategies on top-k chunk retrieval. Helps you select the best pooling method for retrieval accuracy.",
        help="Visualisez l'impact des différentes stratégies de pooling sur la récupération des meilleurs chunks. Permet de choisir la méthode la plus adaptée pour la précision de la recherche."
    )
    chunk_list4 = st.text_area("Or paste chunks (one per line)", chunks_example, key="retrieval_chunks").splitlines()
    query2 = st.text_input("Query for Retrieval", query_example, key="retrieval_query")
    method3 = st.selectbox("Dimensionality Reduction (Retrieval)", ["pca", "umap"], key="retrieval_method")
    strategy3 = st.selectbox("Pooling Strategy (Retrieval)", model.get_supported_strategies(), key="retrieval_strategy")
    if st.button("Run Retrieval Diagnostics", key="retrieval_btn"):
        if not(chunk_list4 and any(chunk_list4)):
            st.error("Please provide some chunks or text to chunk.")
        if not query2:
            st.error("Please provide a query for retrieval.")
        chunks4 = [c for c in chunk_list4 if c.strip()]
        retriever = ChunkRetriever(model)
        retriever.add_chunks(chunks4)
        st.subheader("Pooling Comparison Table")
        #compare_retrieval_pooling(retriever, query2)
        with io.StringIO() as buf, redirect_stdout(buf):
            compare_retrieval_pooling(retriever, query2)
            retrieval_output = buf.getvalue()
        st.text(retrieval_output)
        st.subheader("Chunk Geometry with Query")
        result = plot_chunk_geometry(
            chunks4, model, strategy=strategy3, method=method3, query=query2,
            generate_explanation=generate_explanation, api_key=api_key, language=language
        )
        if generate_explanation and isinstance(result, tuple):
            fig, explanation = result
            st.pyplot(fig)
            st.info(explanation)
        else:
            st.pyplot(result)

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ by SmartBibl.IA. [GitHub](https://github.com/your-repo)")