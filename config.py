import os

# --- General Application Settings ---
DEFAULT_OUTPUT_DIR = "math_json_output"  # Default directory for saving processed JSON files
DEFAULT_APP_DATA_DIR = os.path.join(os.getcwd(), ".polymath_parser_cache") # Base for cache
LOG_LEVEL = "INFO" # Default logging level for the application

# --- PDF Processing Agent Settings ---
# Mathpix API credentials should be set in .env as MATHPIX_APP_ID and MATHPIX_APP_KEY
MD_CACHE_SUBDIR = "md_cache" # Subdirectory for cached Markdown files from Mathpix

# --- Math Text Processor Settings ---
TOKENIZER_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # For chunking and embeddings
CHUNK_MAX_TOKENS = 384  # Max tokens per chunk for LLM processing
MIN_CHUNK_LENGTH_CHARS = 50 # Minimum characters for a text chunk to be considered valid

# --- Frontier Agent (Vector Database / RAG) Settings ---
DB_PATH = "math_vectorstore" # Path to the ChromaDB persistent storage
DB_COLLECTION_NAME = "math_concepts_v4" # ChromaDB collection name
# Frontier agent will try to use GOOGLE_API_KEY, DEEPSEEK_API_KEY, or OPENAI_API_KEY from .env for its own LLM tasks
FRONTIER_AGENT_PREFERRED_MODELS = { # Order of preference if multiple API keys are found
    "Google Gemini": "gemini-1.5-pro-latest",
    "DeepSeek": "deepseek-chat",
    "OpenAI": "gpt-4o-mini"
}
FRONTIER_AGENT_LLM_MAX_TOKENS = 500 # Max tokens for Frontier Agent's internal LLM calls

# --- Specialist Agent (LLM for Concept Extraction) Settings ---
# API Keys for these providers should be in .env (e.g., OPENAI_API_KEY, GOOGLE_API_KEY, etc.)
# The MODEL_CONFIG defines which models are available in the UI and how they map to providers/API details.
# Keys are UI display names.
# 'provider': Maps to the logic in SpecialistAgent for client initialization.
# 'api_model_name': The actual model name string to be used in API calls.
# 'base_url' (optional): For OpenAI-compatible APIs that are not OpenAI directly.
# 'api_key_env' (optional): Specific environment variable for the API key if not the default for the provider.

MODEL_CONFIG = {
    # OpenAI
    "gpt-4o-mini": {"provider": "OpenAI", "api_model_name": "gpt-4o-mini"},
    "gpt-4o": {"provider": "OpenAI", "api_model_name": "gpt-4o"},
    # Google Gemini
    "gemini-1.5-pro-latest": {"provider": "Google Gemini", "api_model_name": "gemini-1.5-pro-latest"},
    "gemini-1.5-flash-latest": {"provider": "Google Gemini", "api_model_name": "gemini-1.5-flash-latest"},
    # DeepSeek
    "deepseek-chat": {"provider": "DeepSeek", "api_model_name": "deepseek-chat", "base_url": "https://api.deepseek.com"},
    "deepseek-coder": {"provider": "DeepSeek", "api_model_name": "deepseek-coder", "base_url": "https://api.deepseek.com"},
    # Anthropic
    "claude-3.5-sonnet-20240620": {"provider": "Anthropic", "api_model_name": "claude-3-5-sonnet-20240620"},
    "claude-3-opus-20240229": {"provider": "Anthropic", "api_model_name": "claude-3-opus-20240229"},
    "claude-3-haiku-20240307": {"provider": "Anthropic", "api_model_name": "claude-3-haiku-20240307"},
    # Cohere
    "command-r-plus": {"provider": "Cohere", "api_model_name": "command-r-plus"},
    "command-r": {"provider": "Cohere", "api_model_name": "command-r"},
    # Mistral - Example: Ensure MISTRAL_API_BASE and MISTRAL_API_KEY are in your .env
    "mistral-large-latest": {"provider": "Mistral", "api_model_name": "mistral-large-latest", "base_url_env": "MISTRAL_API_BASE", "api_key_env": "MISTRAL_API_KEY"},
    "open-mixtral-8x7b": {"provider": "Mistral", "api_model_name": "open-mixtral-8x7b", "base_url_env": "MISTRAL_API_BASE", "api_key_env": "MISTRAL_API_KEY"},
    # Meta Llama - Example: Ensure LLAMA_API_BASE and LLAMA_API_KEY are in .env (e.g., if using TogetherAI or Groq)
    "Llama-3.1-70B-Instruct": {"provider": "Meta (Llama)", "api_model_name": "meta-llama/Llama-3.1-70b-chat-hf", "base_url_env": "LLAMA_API_BASE", "api_key_env": "LLAMA_API_KEY"},
    "Llama-3.1-8B-Instruct": {"provider": "Meta (Llama)", "api_model_name": "meta-llama/Llama-3.1-8b-chat-hf", "base_url_env": "LLAMA_API_BASE", "api_key_env": "LLAMA_API_KEY"},
    # Add other models as needed (Qwen, Phi, etc.) following a similar pattern
}

# Dynamically get base_urls from environment variables if specified
for model_key, model_details in MODEL_CONFIG.items():
    if "base_url_env" in model_details:
        model_details["base_url"] = os.getenv(model_details["base_url_env"])
    # API keys themselves are fetched directly within the SpecialistAgent using os.getenv and the 'api_key_env' or provider defaults.

UI_LLM_MODEL_CHOICES = sorted(list(MODEL_CONFIG.keys())) # For Gradio UI dropdown

SPECIALIST_LLM_MAX_TOKENS = 4096 # Default max tokens for specialist LLM calls (can be model-specific if needed)
SPECIALIST_LLM_TEMPERATURE = 0.05 # Default temperature for specialist LLM calls

# --- UI Settings (polymath_parser_ui.py) ---
LOG_DISPLAY_HEIGHT = "400px" # Min height for the log display box in Gradio
MAX_LOG_LINES_IN_MEMORY = 200 # Max log lines to keep in UI state for display

# --- Function to load API keys (example, actual loading happens in agents) ---
def get_api_key(env_var_name: str) -> str | None:
    """Helper to load API key from environment variables."""
    return os.getenv(env_var_name)

# Example of how you might structure API key names if you wanted to centralize the names
# The actual os.getenv calls would still happen where the keys are needed.
# API_KEY_NAMES = {
# "openai": "OPENAI_API_KEY",
# "google_gemini": "GOOGLE_API_KEY",
# "anthropic": "ANTHROPIC_API_KEY",
# "cohere": "COHERE_API_KEY",
# "deepseek": "DEEPSEEK_API_KEY",
# "mathpix_id": "MATHPIX_APP_ID",
# "mathpix_key": "MATHPIX_APP_KEY",
# "mistral": "MISTRAL_API_KEY", # if using a dedicated key for mistral
# "llama_meta": "LLAMA_API_KEY" # if using a dedicated key for llama via a provider
# }