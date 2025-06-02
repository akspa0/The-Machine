import argparse
from pathlib import Path
import sys
import hashlib
import json

# Try to import tiktoken for token-based chunking
try:
    import tiktoken
    def tokenize_text(text, model="gpt-3.5-turbo"):
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding('cl100k_base')
        return enc.encode(text), enc
except ImportError:
    tiktoken = None
    def tokenize_text(text, model=None):
        # Fallback: 1 token ≈ 4 chars
        return list(text), None

def split_into_chunks(text, max_tokens=3500, model="gpt-3.5-turbo", overlap=0, context_seed=None):
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
        chunks = split_into_chunks(current_text, max_tokens=chunk_size, model=model, overlap=overlap, context_seed=context_seed)
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
    from llm_utils import run_llm_task
    if prompt_template is None:
        prompt = "Summarize the following text as concisely as possible.\n\n"
    else:
        prompt = prompt_template + "\n\n"
    for i, chunk in enumerate(chunks):
        prompt += f"Chunk {i+1}:\n{chunk}\n\n"
    summary = run_llm_task(prompt, llm_config, single_output=True, chunking=False)
    return summary.strip()

def main():
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
        chunks = split_into_chunks(text, max_tokens=args.chunk_size, model=args.model, overlap=args.overlap, context_seed=context_seed)
        result = '\n\n'.join(chunks)
    if args.output_file:
        Path(args.output_file).write_text(result, encoding='utf-8')
        print(f'[INFO] Wrote result to {args.output_file}')
    else:
        print(result)

if __name__ == '__main__':
    main() 