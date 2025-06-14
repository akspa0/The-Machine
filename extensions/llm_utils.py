import requests
import json
import hashlib
from pathlib import Path
import subprocess
from typing import List

# Tokenization utility for chunking
try:
    import tiktoken
    def split_into_chunks(text, max_tokens=3500, model="gpt-3.5-turbo"):
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            # Unknown model; default to cl100k_base silently
            enc = tiktoken.get_encoding("cl100k_base")
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


def run_llm_task(prompt, config, output_path=None, seed=None, chunking=True, single_output=False, system_prompt=None):
    """
    Run an LLM task using the config dict (same as pipeline_orchestrator.py).
    If chunking=True and the prompt is too long, split into chunks and aggregate results.
    If single_output=True, only use the first chunk (for title, etc.).
    Returns the LLM response as a string. Optionally writes to output_path.
    No continuation or sliding window logic: each chunk is sent as a single prompt, and only the direct response is used.
    """
    base_url = config.get('lm_studio_base_url', 'http://localhost:1234/v1')
    api_key = config.get('lm_studio_api_key', 'lm-studio')
    model_id = config.get('lm_studio_model_identifier', 'l3-grand-horror-ii-darkest-hour-uncensored-ed2.15-15b')
    temperature = config.get('lm_studio_temperature', 0.5)
    max_tokens = config.get('lm_studio_max_tokens', 8192)
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    results = []
    if chunking:
        chunks = split_into_chunks(prompt, max_tokens=3500, model=model_id)
        if len(chunks) > 1:
            print(f"[INFO] Splitting prompt into {len(chunks)} chunks for LLM task.")
        if single_output:
            chunks = [chunks[0]]
    else:
        chunks = [prompt]
    for idx, chunk in enumerate(chunks):
        chunk_seed = seed
        if chunk_seed is None:
            chunk_seed = int(hashlib.sha256((chunk + str(idx)).encode()).hexdigest(), 16) % (2**32)
        data = {
            "model": model_id,
            "messages": [],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": chunk_seed
        }
        if system_prompt:
            data["messages"].append({"role": "system", "content": system_prompt})
        data["messages"].append({"role": "user", "content": chunk})
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


def load_lm_studio_model(path=None, context_length=4096, identifier=None, yes=True, gpu=None, ttl=None, log_level=None, host=None, port=None):
    """
    Load an LM Studio model with the specified context length and options using the lms CLI.
    Returns True if successful, False otherwise.
    """
    cmd = ["lms", "load"]
    if path:
        cmd.append(str(path))
    if context_length:
        cmd += ["--context-length", str(context_length)]
    if identifier:
        cmd += ["--identifier", str(identifier)]
    if yes:
        cmd.append("--yes")
    if gpu:
        cmd += ["--gpu", str(gpu)]
    if ttl:
        cmd += ["--ttl", str(ttl)]
    if log_level:
        cmd += ["--log-level", str(log_level)]
    if host:
        cmd += ["--host", str(host)]
    if port:
        cmd += ["--port", str(port)]
    print(f"[INFO] Loading LM Studio model with command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        print(f"[INFO] lms load stdout: {result.stdout.strip()}")
        if result.stderr:
            print(f"[WARN] lms load stderr: {result.stderr.strip()}")
        if result.returncode != 0:
            print(f"[ERROR] lms load command exited with code {result.returncode}")
            return False
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load LM Studio model: {e}")
        return False


def unload_lm_studio_model(identifier=None, host=None, port=None):
    """
    Unload an LM Studio model by identifier (or all if not specified) using the lms CLI.
    Returns True if successful, False otherwise.
    """
    cmd = ["lms", "unload"]
    if identifier:
        cmd += ["--identifier", str(identifier)]
    if host:
        cmd += ["--host", str(host)]
    if port:
        cmd += ["--port", str(port)]
    print(f"[INFO] Unloading LM Studio model with command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        print(f"[INFO] lms unload stdout: {result.stdout.strip()}")
        if result.stderr:
            print(f"[WARN] lms unload stderr: {result.stderr.strip()}")
        if result.returncode != 0:
            print(f"[ERROR] lms unload command exited with code {result.returncode}")
            return False
        return True
    except Exception as e:
        print(f"[ERROR] Failed to unload LM Studio model: {e}")
        return False

# Advanced tokenization for chunking (with tiktoken fallback)
def tokenize_text(text, model="gpt-3.5-turbo"):
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding('cl100k_base')
        return enc.encode(text), enc
    except ImportError:
        # Fallback: 1 token ≈ 4 chars
        return list(text), None

def split_into_chunks_advanced(text, max_tokens=3500, model="gpt-3.5-turbo", overlap=0, context_seed=None):
    """
    Split text into token-limited chunks with optional overlap and context seed prepended to each chunk.
    """
    tokens, enc = tokenize_text(text, model)
    chunks = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        if enc:
            chunk_text = enc.decode(chunk_tokens)
        else:
            chunk_text = ''.join(chunk_tokens)
        if context_seed:
            chunk_text = context_seed + '\n' + chunk_text
        chunks.append(chunk_text)
        if overlap > 0:
            i += max_tokens - overlap
        else:
            i += max_tokens
    return chunks

def recursive_summarize(text, llm_summarize_fn, max_chunks=8, chunk_size=3500, overlap=0, context_seed=None, model="gpt-3.5-turbo", llm_config=None, prompt_template=None, verbose=False):
    """
    Recursively chunk and summarize text until the number of chunks is <= max_chunks.
    llm_summarize_fn: function(chunks, llm_config, prompt_template) -> summary
    """
    current_text = text
    while True:
        chunks = split_into_chunks_advanced(current_text, max_tokens=chunk_size, model=model, overlap=overlap, context_seed=context_seed)
        if verbose:
            print(f"[INFO] Chunks: {len(chunks)} (max allowed: {max_chunks})")
        if len(chunks) <= max_chunks:
            return chunks
        # Summarize all chunks into a single text
        summary = llm_summarize_fn(chunks, llm_config, prompt_template=prompt_template)
        if verbose:
            print(f"[INFO] Summarized {len(chunks)} chunks into {len(summary)} chars.")
        current_text = summary

def default_llm_summarize_fn(chunks, llm_config, prompt_template=None):
    """
    Default LLM summarization using run_llm_task from llm_utils.
    """
    prompt = (prompt_template + "\n\n") if prompt_template else "Summarize the following text as concisely as possible.\n\n"
    for i, chunk in enumerate(chunks):
        prompt += f"Chunk {i+1}:\n{chunk}\n\n"
    summary = run_llm_task(prompt, llm_config, single_output=True, chunking=False)
    return summary.strip()

# CLI entry point for chunking/summarization
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="LLM Chunker: Token-based chunking and recursive summarization utility.")
    parser.add_argument('--input-file', type=str, required=True, help='Input text file to chunk/summarize.')
    parser.add_argument('--output-file', type=str, help='Output file for final result (default: stdout).')
    parser.add_argument('--llm-config', type=str, help='Path to LLM config JSON (for summarization).')
    parser.add_argument('--chunk-size', type=int, default=3500, help='Max tokens per chunk (default: 3500).')
    parser.add_argument('--overlap', type=int, default=0, help='Token overlap between chunks (default: 0).')
    parser.add_argument('--context-seed', type=str, help='Text to prepend to each chunk.')
    parser.add_argument('--recursive', action='store_true', help='Recursively summarize until under max-chunks.')
    parser.add_argument('--max-chunks', type=int, default=8, help='Max chunks before summarizing recursively (default: 8).')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='Model name for tokenization (default: gpt-3.5-turbo).')
    parser.add_argument('--prompt-template', type=str, help='Custom prompt template for summarization.')
    parser.add_argument('--verbose', action='store_true', help='Verbose output.')
    args = parser.parse_args()

    text = Path(args.input_file).read_text(encoding='utf-8')
    context_seed = args.context_seed
    if args.recursive:
        if not args.llm_config:
            print('[ERROR] --llm-config is required for recursive summarization.')
            sys.exit(1)
        llm_config = json.load(open(args.llm_config, 'r', encoding='utf-8'))
        prompt_template = args.prompt_template
        chunks = recursive_summarize(
            text,
            default_llm_summarize_fn,
            max_chunks=args.max_chunks,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            context_seed=context_seed,
            model=args.model,
            llm_config=llm_config,
            prompt_template=prompt_template,
            verbose=args.verbose
        )
        result = '\n\n'.join(chunks)
    else:
        chunks = split_into_chunks_advanced(text, max_tokens=args.chunk_size, model=args.model, overlap=args.overlap, context_seed=context_seed)
        result = '\n\n'.join(chunks)
    if args.output_file:
        Path(args.output_file).write_text(result, encoding='utf-8')
        print(f'[INFO] Wrote result to {args.output_file}')
    else:
        print(result)

# ---------------------------------------------------------------------------
#  Simple LLM Task Manager (pipeline-friendly helper)
# ---------------------------------------------------------------------------


class LLMTaskManager:
    """Queue and run multiple LLM tasks in sequence.

    This lightweight helper is *stateless* – results are returned to the caller
    but you can also optionally persist each task's output to a file for
    downstream extensions.
    """

    def __init__(self, llm_config: dict):
        self.llm_config = llm_config
        self.tasks = []  # list of (prompt, kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, prompt: str, *, output_path: str | Path | None = None, **kwargs):
        """Queue a prompt for later execution.

        Additional **kwargs are forwarded to `run_llm_task` (e.g. single_output,
        chunking, system_prompt, seed, etc.).
        """
        self.tasks.append((prompt, dict(output_path=output_path, **kwargs)))

    def run_all(self) -> List[str]:
        """Execute all queued tasks and return list of results in order."""
        results: List[str] = []
        for prompt, kw in self.tasks:
            res = run_llm_task(prompt, self.llm_config, **kw)
            results.append(res)
        return results

    # Convenience classmethod ------------------------------------------------

    @classmethod
    def run(cls, tasks: List[dict], llm_config: dict) -> List[str]:
        """Fire-and-forget helper.

        *tasks* is a list of dicts with at minimum a ``prompt`` key and any
        optional arguments accepted by :func:`run_llm_task`.
        """
        mgr = cls(llm_config)
        for task in tasks:
            prompt = task.pop("prompt")
            mgr.add(prompt, **task)
        return mgr.run_all() 