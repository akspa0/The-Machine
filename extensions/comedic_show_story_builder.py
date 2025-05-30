import re
import json
from pathlib import Path
from extension_base import ExtensionBase
from llm_utils import run_llm_task

def parse_timeline(show_txt_path):
    """Parse completed-show.txt to extract call titles and timestamps."""
    timeline = []
    with open(show_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = re.match(r'^🎙️ (.+) \((\d+:\d+:\d+) - (\d+:\d+:\d+)\)', line.strip())
            if m:
                title, start, end = m.groups()
                timeline.append({'title': title, 'start': start, 'end': end})
    return timeline

def load_synopsis(llm_dir, call_id):
    synopsis_path = llm_dir / call_id / 'call_synopsis.txt'
    if synopsis_path.exists():
        return synopsis_path.read_text(encoding='utf-8').strip()
    return None

def load_transcript_excerpt(calls_dir, call_title):
    # Try to load a short excerpt from the transcript (first non-empty line)
    transcript_path = calls_dir / f'{call_title}_transcript.txt'
    if transcript_path.exists():
        with open(transcript_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    return line[:120]  # Truncate to 120 chars
    return None

def paraphrase_moment(title, start, synopsis, excerpt):
    # Combine synopsis and excerpt for a vivid, detail-rich moment
    if synopsis and excerpt:
        base = f"{start}: {synopsis} " + f"Quote: '{excerpt}'"
    elif synopsis:
        base = f"{start}: {synopsis}"
    elif excerpt:
        base = f"{start}: '{excerpt}'"
    else:
        base = f"{start}: The story took an unexpected turn at this point."
    # Remove any call title references
    base = re.sub(r"'[^']+'", '', base)
    return base

STORY_STYLES = {
    'comedic': {
        'filename': 'comedic_show_story.txt',
        'prompt': (
            "Given the following sequence of show moments, each with a timestamp and a brief, specific summary (without call titles), write a single, funny, story-like paragraph that chronicles the entire show in order. "
            "Be creative and story-like, but focus on the details provided in each moment. Weave together the events and moods, referencing the timestamp at each transition or key moment. Do not use or reference call titles. Be concise, punchy, and privacy-safe."
        )
    },
    'dramatic': {
        'filename': 'dramatic_show_story.txt',
        'prompt': (
            "Given the following sequence of show moments, each with a timestamp and a brief, specific summary (without call titles), write a single, dramatic, suspenseful paragraph that chronicles the entire show in order. "
            "Be creative and story-like, but focus on the details provided in each moment. Weave together the events and moods, referencing the timestamp at each transition or key moment. Do not use or reference call titles. Be vivid, intense, and privacy-safe."
        )
    },
    'confusing': {
        'filename': 'confusing_show_story.txt',
        'prompt': (
            "Given the following sequence of show moments, each with a timestamp and a brief, specific summary (without call titles), write a single, surreal, confusing, dreamlike paragraph that chronicles the entire show in order. "
            "Be creative and story-like, but focus on the details provided in each moment. Weave together the events and moods, referencing the timestamp at each transition or key moment. Do not use or reference call titles. Be intentionally bewildering, but privacy-safe."
        )
    }
}

class ComedicShowStoryBuilder(ExtensionBase):
    def run(self):
        self.log("Starting multi-style show story generation.")
        show_txt = self.output_root / 'finalized' / 'show' / 'completed-show.txt'
        llm_dir = self.output_root / 'llm'
        calls_dir = self.output_root / 'finalized' / 'calls'
        if not show_txt.exists():
            self.log(f"Show file not found: {show_txt}")
            return
        timeline = parse_timeline(show_txt)
        if not timeline:
            self.log("No calls found in show timeline.")
            return
        # Try to map call titles to call_ids using manifest
        call_title_to_id = {}
        if self.manifest:
            for entry in self.manifest:
                if entry.get('stage') == 'remix':
                    call_id = entry.get('call_id')
                    call_title_path = llm_dir / call_id / 'call_title.txt'
                    if call_title_path.exists():
                        title = call_title_path.read_text(encoding='utf-8').splitlines()[0].strip().strip('"')
                        call_title_to_id[title] = call_id
        # Build moments
        moments = []
        for call in timeline:
            title = call['title']
            start = call['start']
            call_id = call_title_to_id.get(title)
            synopsis = load_synopsis(llm_dir, call_id) if call_id else None
            excerpt = load_transcript_excerpt(calls_dir, title)
            moment = paraphrase_moment(title, start, synopsis, excerpt)
            moments.append(moment)
        # Truncate if needed for token limit
        max_moments = 40  # Adjust as needed for token budget
        if len(moments) > max_moments:
            moments = moments[:max_moments]
        # Load LLM config
        llm_config_path = self.output_root / 'workflows' / 'llm_tasks.json'
        if llm_config_path.exists():
            config_used = llm_config_path
        else:
            project_root = Path(__file__).resolve().parent.parent
            root_config = project_root / 'workflows' / 'llm_tasks.json'
            if root_config.exists():
                llm_config_path = root_config
                config_used = root_config
            else:
                self.log(f"LLM config not found in {self.output_root / 'workflows'} or {root_config}")
                return
        self.log(f"Using LLM config: {llm_config_path}")
        with open(llm_config_path, 'r', encoding='utf-8') as f:
            llm_config = json.load(f)
        # Generate and save each story style
        for style, meta in STORY_STYLES.items():
            out_path = self.output_root / 'finalized' / 'show' / meta['filename']
            prompt = (
                meta['prompt'] + "\n\nMoments:\n" + '\n'.join(f"- {m}" for m in moments)
            )
            story = run_llm_task(prompt, llm_config, output_path=out_path)
            self.log(f"{style.capitalize()} show story written to {out_path}")
            self.log(f"{style.capitalize()} preview: {story[:300]}...")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python comedic_show_story_builder.py <output_root>")
        exit(1)
    ext = ComedicShowStoryBuilder(sys.argv[1])
    ext.run() 