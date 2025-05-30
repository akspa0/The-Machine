import requests
import json
import hashlib
from pathlib import Path

# Tokenization utility for chunking
try:
    import tiktoken
    def split_into_chunks(text, max_tokens=3500, model="gpt-3.5-turbo"):
        enc = tiktoken.encoding_for_model(model)
        tokens = enc.encode(text)
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i+max_tokens]
            chunk_text = enc.decode(chunk_tokens)
            chunks.append(chunk_text)
        return chunks
except ImportError:
    def split_into_chunks(text, max_tokens=3500, model=None):
        # Fallback: 1 token ≈ 4 chars
        max_chars = max_tokens * 4
        return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]


def run_llm_task(prompt, config, output_path=None, seed=None, chunking=True, single_output=False):
    """
    Run an LLM task using the config dict (same as pipeline_orchestrator.py).
    If chunking=True and the prompt is too long, split into chunks and aggregate results.
    If single_output=True, only use the first chunk (for title, etc.).
    Returns the LLM response as a string. Optionally writes to output_path.
    """
    base_url = config.get('lm_studio_base_url', 'http://localhost:1234/v1')
    api_key = config.get('lm_studio_api_key', 'lm-studio')
    model_id = config.get('lm_studio_model_identifier', 'llama-3.1-8b-supernova-etherealhermes')
    temperature = config.get('lm_studio_temperature', 0.5)
    max_tokens = config.get('lm_studio_max_tokens', 2048)
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    # Chunking logic
    results = []
    if chunking:
        chunks = split_into_chunks(prompt, max_tokens=3500, model=model_id)
        if len(chunks) > 1:
            print(f"[INFO] Splitting prompt into {len(chunks)} chunks for LLM task.")
        # For single_output tasks (e.g., title), only use the first chunk
        if single_output:
            chunks = [chunks[0]]
    else:
        chunks = [prompt]
    for idx, chunk in enumerate(chunks):
        # Deterministic seed if provided
        chunk_seed = seed
        if chunk_seed is None:
            chunk_seed = int(hashlib.sha256((chunk + str(idx)).encode()).hexdigest(), 16) % (2**32)
        data = {
            "model": model_id,
            "messages": [
                {"role": "user", "content": chunk}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": chunk_seed
        }
        try:
            response = requests.post(f"{base_url}/chat/completions", headers=headers, data=json.dumps(data), timeout=60)
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                results.append(content)
            else:
                error_msg = f"LLM API error {response.status_code}: {response.text}"
                results.append(error_msg)
        except Exception as e:
            fail_msg = f"LLM request failed: {e}"
            results.append(fail_msg)
    final_result = '\n\n'.join(results)
    if output_path:
        Path(output_path).write_text(final_result, encoding='utf-8')
    return final_result 