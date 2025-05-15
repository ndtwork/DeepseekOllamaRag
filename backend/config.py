# config.py

MODEL_CONFIG = {
    "embedding_backend": "ollama",      # "huggingface", "ollama", ...
    "embedding_model": "nomic-embed-text",  # hoặc "sentence-transformers/all-MiniLM-L6-v2"

    "llm_backend": "ollama",            # "huggingface", "ollama", "openai"
    "llm_model": "mistral"              # hoặc "llama3", "gpt-4",...
}
