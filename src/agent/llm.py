"""LLM factory. Default: local Ollama (Docker container on localhost:11434)."""
from __future__ import annotations


def get_llm(cfg: dict, **overrides):
    c = {**cfg.get('llm', {}), **overrides}
    provider = c.get('provider', 'ollama')
    if provider == 'ollama':
        from langchain_ollama import ChatOllama
        return ChatOllama(model=c['model'], base_url=c.get('base_url', 'http://localhost:11434'),
                          temperature=c.get('temperature', 0.0), reasoning=c.get('think', False),
                          num_ctx=c.get('num_ctx', 8192), keep_alive=c.get('keep_alive', '30m'),
                          num_predict=c.get('num_predict', 1024), seed=c.get('seed'))
    if provider == 'fake':
        # Used by tests: overrides must include 'responses' (list of pydantic objects / strings)
        return c['instance']
    raise ValueError(f'Unknown LLM provider {provider!r}')


def structured(llm, schema):
    """Structured output; json_schema is the most reliable mode for Ollama models."""
    try:
        return llm.with_structured_output(schema, method='json_schema')
    except TypeError:
        return llm.with_structured_output(schema)
