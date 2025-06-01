import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from llm_utils import run_llm_task

def summarize_chunks_with_llm(chunks, llm_config):
    prompt = (
        "Summarize the following persona descriptions into a single, creative, visually rich SDXL prompt for a character portrait. "
        "Favor creativity, absurd humor, and surreal situations.\n\n"
    )
    for i, chunk in enumerate(chunks):
        prompt += f"Chunk {i+1}:\n{chunk}\n\n"
    summary = run_llm_task(prompt, llm_config, single_output=True, chunking=False)
    return summary.strip()

def main():
    parser = argparse.ArgumentParser(description='Summarize multiple text chunks into a single SDXL prompt using the LLM.')
    parser.add_argument('--input-files', type=str, nargs='+', required=True, help='Input chunk text files.')
    parser.add_argument('--output', type=str, required=True, help='Output file for summary.')
    parser.add_argument('--llm-config', type=str, default='workflows/llm_tasks.json', help='Path to LLM config JSON.')
    args = parser.parse_args()
    chunks = [Path(f).read_text(encoding='utf-8') for f in args.input_files]
    import json
    llm_config = json.load(open(args.llm_config, 'r', encoding='utf-8'))
    summary = summarize_chunks_with_llm(chunks, llm_config)
    Path(args.output).write_text(summary, encoding='utf-8')
    print(f'[INFO] Wrote summary to {args.output}')

if __name__ == '__main__':
    main() 