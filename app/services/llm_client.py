from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from .utils import get_env
from config_base import (
    DEFAULT_LLM_MODEL,
    FALLBACK_LLM_MODEL,
    OPENAI_LLM_MODEL,
    GOOGLE_LLM_MODEL,
    GROQ_LLM_MODEL,
)

OPENROUTER_API_KEY = get_env("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = get_env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENAI_API_KEY = get_env("OPENAI_API_KEY")
GOOGLEAI_API_KEY = get_env("GOOGLEAI_API_KEY")
GROQ_API_KEY = get_env("GROQ_API_KEY")
GROQ_BASE_URL = get_env("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

# ============================================================
# 1) CLIENTE SIMPLE (async, sin LangChain)
# ============================================================

# Cliente minimalista para prompts directos sin LangChain. Devuelve solo texto. Ideal para endpoints simples.
# ---------- OpenRouter ----------
async def llm(prompt: str, model: str | None = None) -> str:
    if not OPENROUTER_API_KEY or not OPENROUTER_BASE_URL:
        raise ValueError("OPENROUTER_API_KEY o OPENROUTER_BASE_URL no están configuradas.")
    
    # Cliente OpenRouter compatible con AsyncOpenAI
    client = AsyncOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )
    
    model_to_use = model or DEFAULT_LLM_MODEL
    
    params = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 400,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    
    for model in [model_to_use, FALLBACK_LLM_MODEL]:
        try:
            response = await client.chat.completions.create(model=model, **params)
            return response.choices[0].message.content
        except Exception as e:
            print(f"[WARN] Modelo {model} falló: {e}")
    
    raise RuntimeError("No se pudo obtener respuesta de ningún modelo.")

# ---------- Groq ----------
async def llm_groq(prompt: str, model: str | None = None) -> str:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY no está configurada.")

    client = AsyncOpenAI(
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
    )
    
    model_to_use = model or GROQ_LLM_MODEL

    response = await client.chat.completions.create(
        model=model_to_use,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.7,
        top_p=0.9,
    )

    return response.choices[0].message.content

# ============================================================
# 2) CLIENTES LANGCHAIN (LLMChain, agentes, tools, etc.)
# ============================================================

# Devuelve un objeto ChatOpenAI configurado para OpenRouter. Compatible con LLMChain, RouterChain, MultiPromptChain, agentes, etc.
# ---------- OpenRouter ----------
def llm_chain(model: str | None = None, temperature: float = 0.0) -> ChatOpenAI:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY no está configurada.")
    
    model_to_use = model or DEFAULT_LLM_MODEL

    llm_params = {
        "api_key": OPENROUTER_API_KEY,
        "base_url": OPENROUTER_BASE_URL,
        "temperature": temperature,
    }

    for model in [model_to_use, FALLBACK_LLM_MODEL]:
        try:
            return ChatOpenAI(model=model, **llm_params)
        except Exception as e:
            print(f"[WARN] Falló al crear LLM con modelo '{model}': {e}")

    raise RuntimeError("Ningún modelo pudo ser inicializado.")

# ---------- Groq ----------
def llm_chain_groq(model: str | None = None, temperature: float = 0.0) -> ChatOpenAI:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY no está configurada.")

    model_to_use = model or GROQ_LLM_MODEL
    
    try:
        return ChatOpenAI(
            model=model_to_use,
            api_key=GROQ_API_KEY,
            base_url=GROQ_BASE_URL,
            temperature=temperature,
        )
    except Exception as e:
        raise RuntimeError(
            f"Error inicializando Groq LLM '{model}'."
        ) from e
    
# Devuelve un objeto ChatOpenAI configurado para OpenAI. Compatible con LLMChain, RouterChain, MultiPromptChain, agentes, etc.
# ---------- OpenAI ----------
def llm_chain_openai(model: str | None = None, temperature: float = 0.7,) -> ChatOpenAI:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY no está configurada.")
    
    model_to_use = model or OPENAI_LLM_MODEL

    llm_params = {
        "api_key": OPENAI_API_KEY,
        "temperature": temperature,
    }
    
    try:
        return ChatOpenAI(model=model_to_use, **llm_params)
    except Exception as e:
        raise RuntimeError(f"Error inicializando ChatOpenAI con el modelo '{model_to_use}'.") from e

# Devuelve un objeto ChatGoogleGenerativeAI configurado para Google Generative AI. Compatible con LLMChain, RouterChain, MultiPromptChain, agentes, etc.
# ---------- Google ----------
def llm_chain_google(model: str | None = None, temperature: float = 0.7,) -> ChatGoogleGenerativeAI:
    if not GOOGLEAI_API_KEY:
        raise ValueError("GOOGLEAI_API_KEY no está configurada.")
    
    model_to_use = model or GOOGLE_LLM_MODEL

    llm_params = {
        "api_key": GOOGLEAI_API_KEY,
        "temperature": temperature,
    }

    try:
        return ChatGoogleGenerativeAI(model=model_to_use, **llm_params)
    except Exception as e:
        raise RuntimeError(f"No se pudo inicializar el modelo Google '{model_to_use}'.") from e
