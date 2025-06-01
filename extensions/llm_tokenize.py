import argparse
from pathlib import Path
import sys

def split_text_to_token_chunks(text, max_tokens=4096, model='gpt-3.5-turbo'):
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            print(f'[WARN] tiktoken: Unknown model "{model}", falling back to cl100k_base encoding.')
            enc = tiktoken.get_encoding('cl100k_base')
    except ImportError:
        print('[ERROR] tiktoken is required for tokenization.')
        sys.exit(1)
    tokens = enc.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

def main():
    parser = argparse.ArgumentParser(description='Split input text into token-limited chunks using tiktoken.')
    parser.add_argument('--input-file', type=str, required=True, help='Input text file to chunk.')
    parser.add_argument('--max-tokens', type=int, default=4096, help='Max tokens per chunk (default: 4096).')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='Model name for tiktoken encoding (default: gpt-3.5-turbo).')
    parser.add_argument('--output-dir', type=str, help='Directory to write chunk files. If not set, print to stdout.')
    args = parser.parse_args()
    text = Path(args.input_file).read_text(encoding='utf-8')
    chunks = split_text_to_token_chunks(text, max_tokens=args.max_tokens, model=args.model)
    if args.output_dir:
        outdir = Path(args.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        for i, chunk in enumerate(chunks):
            chunk_path = outdir / f'chunk_{i+1:03d}.txt'
            chunk_path.write_text(chunk, encoding='utf-8')
            print(f'[INFO] Wrote {chunk_path}')
    else:
        for i, chunk in enumerate(chunks):
            print(f'--- Chunk {i+1} ---\n{chunk}\n')

if __name__ == '__main__':
    main() 