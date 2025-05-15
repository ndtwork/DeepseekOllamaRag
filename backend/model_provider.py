# model_provider.py

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OllamaEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFacePipeline
from langchain_community.llms import Ollama
from config import MODEL_CONFIG


def get_embedding_model():
    if MODEL_CONFIG["embedding_backend"] == "huggingface":
        return HuggingFaceEmbeddings(model_name=MODEL_CONFIG["embedding_model"])
    elif MODEL_CONFIG["embedding_backend"] == "ollama":
        return OllamaEmbeddings(model=MODEL_CONFIG["embedding_model"])
    else:
        raise ValueError("Unsupported embedding backend")


def get_llm():
    if MODEL_CONFIG["llm_backend"] == "huggingface":
        return HuggingFacePipeline.from_model_id(
            model_id=MODEL_CONFIG["llm_model"],
            task="text-generation"
        )
    elif MODEL_CONFIG["llm_backend"] == "ollama":
        return Ollama(model=MODEL_CONFIG["llm_model"])
    elif MODEL_CONFIG["llm_backend"] == "openai":
        return ChatOpenAI(model_name=MODEL_CONFIG["llm_model"])
    else:
        raise ValueError("Unsupported LLM backend")
