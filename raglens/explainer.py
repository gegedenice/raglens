import os
from openai import OpenAI
import base64
import matplotlib.pyplot as plt
import io

def get_base64_encoded_plot(fig=None):
    buf = io.BytesIO()
    if fig is None:
        plt.savefig(buf, format="png", bbox_inches="tight")
    else:
        fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    base64_img = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return base64_img

def generate_plot_explanation(image, format="path", provider="openai", api_key=None, 
                            plot_type=None, text_data=None, additional_context=None, language="English"):
    """
    Generate AI explanation for a plot with enhanced context.
    
    Args:
        image: Base64 encoded image or file path
        format: "base64" or "path"
        provider: "openai" or other providers
        api_key: API key for the provider
        plot_type: Type of plot ("token_geometry", "chunk_geometry", "pooling_comparison", 
                   "semantic_similarity", "embedding_stats", "token_drift", "chunking_length")
        text_data: The text data that was visualized (tokens, chunks, sentences, etc.)
        additional_context: Any additional context about the plot
    """
    if provider == "openai":
        api_key = api_key or os.getenv("OPENAI_API_KEY")   
        client = OpenAI(base_url="https://api.openai.com/v1", api_key=api_key)
        
        if format == "base64":
            img_data = image
        else:
            with open(image, "rb") as f:
                img_data = base64.b64encode(f.read()).decode()

        # Build context-aware prompts
        system_prompt, user_prompt = _build_contextual_prompts(plot_type, text_data, additional_context, language)

        response = client.chat.completions.create(
            model="o4-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_data}"}}
                    ]
                }
            ]
        )
        return response.choices[0].message.content

    else:
        raise NotImplementedError(f"Provider '{provider}' is not yet supported.")

def _build_contextual_prompts(plot_type, text_data, additional_context, language):
    """Build context-aware system and user prompts based on plot type."""
    
    # Base system prompt
    base_system = """You are an expert in machine learning and natural language processing, specializing in analyzing embedding visualizations and diagnostic plots. Your task is to provide clear, technical interpretations of these plots that help users understand their embedding model behavior."""
    
    # Plot-specific system prompts
    plot_system_prompts = {
        "token_geometry": """You are analyzing a token embedding visualization. This plot shows how individual tokens from a text are positioned in 2D space after dimensionality reduction (PCA or UMAP). Each point represents a token, and the spatial relationships reveal semantic similarities and clustering patterns in the embedding space.""",
        
        "chunk_geometry": """You are analyzing a chunk embedding visualization. This plot shows how text chunks (sentences or paragraphs) are positioned in 2D space after dimensionality reduction. Each point represents a chunk, and if a query is included, it's marked in red. This helps understand semantic relationships between chunks and their relevance to queries.""",
        
        "pooling_comparison": """You are analyzing a pooling strategy comparison heatmap. This shows cosine similarities between different pooling methods (CLS, mean, max) applied to the same text. Values close to 1 indicate similar representations, while lower values suggest different semantic interpretations.""",
        
        "semantic_similarity": """You are analyzing a semantic similarity matrix. This heatmap shows pairwise cosine similarities between sentences using different pooling strategies. Values range from 0 (no similarity) to 1 (identical), revealing how well the model captures semantic relationships.""",
        
        "embedding_stats": """You are analyzing embedding distribution statistics. This includes L2 norm distributions and PCA explained variance plots, showing the statistical properties of token embeddings and how much variance is captured by principal components.""",
        
        "token_drift": """You are analyzing layer-wise token drift. This plot shows how token representations change across transformer layers, measured by L2 distance between consecutive layers. This reveals how information flows and transforms through the model.""",
        
        "chunking_length": """You are analyzing chunk length distribution. This bar plot shows the token count for each text chunk, helping assess chunking strategy effectiveness and identify potential issues with chunk sizes."""
    }
    
    # Build user prompt with context
    user_prompt = "Please interpret this diagnostic plot"
    
    if plot_type and plot_type in plot_system_prompts:
        system_prompt = plot_system_prompts[plot_type]
    else:
        system_prompt = base_system
    
    # Add text data context to user prompt
    if text_data:
        if plot_type == "token_geometry":
            user_prompt += f" for the following tokens: {text_data}"
        elif plot_type == "chunk_geometry":
            user_prompt += f" for the following text chunks: {text_data}"
        elif plot_type == "semantic_similarity":
            user_prompt += f" for the following sentences: {text_data}"
        elif plot_type == "token_drift":
            user_prompt += f" tracking these tokens: {text_data}"
    
    # Add additional context
    if additional_context:
        user_prompt += f"\n\nAdditional context: {additional_context}"
    
    user_prompt += ".\n\nPlease provide a detailed analysis including:\n1. What the plot shows\n2. Key patterns or insights\n3. Potential implications for the embedding model\n4. Any concerns or recommendations"
     # Add language instruction
    user_prompt += f"\n\nPlease provide your analysis in {language}."
    
    return system_prompt, user_prompt
