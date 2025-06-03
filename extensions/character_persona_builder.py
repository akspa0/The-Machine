import os
import json
import string
from pathlib import Path
from extension_base import ExtensionBase
from llm_utils import run_llm_task, split_into_chunks_advanced, recursive_summarize, default_llm_summarize_fn
import argparse
from pydub.utils import mediainfo
from pydub import AudioSegment
import soundfile as sf
import numpy as np

"""
Usage:
    python character_persona_builder.py <output_root> [--llm-config <config_path>]

For each call_id in <output_root>/speakers/:
- If left-vocals/right-vocals are present, merges all speakers per channel and generates one persona per channel.
- If only conversation is present, generates a separate persona for each speaker in conversation (no merging).
Handles channel folders named with prefixes (e.g., 0000-conversation) and normalizes for output.
Outputs to <output_root>/characters/<call_title or call_id>/<channel or conversation_speaker>/.
"""

# Distilled Character.AI persona creation guidelines (from character_ai_guidelines.md and official docs)
CHARACTER_AI_GUIDELINES = '''
Character.AI Persona Creation Guidelines:
- Fill out all core attributes: Name, Greeting, Short Description, Long Description, Suggested Starters, Categories.
- Write a detailed Definition: include background, personality, quirks, and example dialogs.
- Use the transcript and call synopsis below to infer the character's traits, style, and context.
- Make the persona engaging, specific, and ready for use in Character.AI.
- Example dialog should demonstrate the character's unique voice and behavior.
- Use best practices: be specific, provide examples, create depth, and consider context.
- Be concise and allow for absurdity to flow naturally in the persona.
- Try to keep the entire response below 300 tokens.
- Output in the following format:

Name: <Character Name>
Short Description: <One-line description>
Greeting: <Greeting message>
Long Description: <Detailed background/personality>
Suggested Starters:
- <Starter 1>
- <Starter 2>
Categories: <comma-separated tags>
Definition:
<Freeform definition, including example dialogs and behavioral notes>
'''

def sanitize_title(title):
    title = title.translate(str.maketrans('', '', string.punctuation))
    title = '_'.join(title.strip().split())
    return title[:48] or None

def create_audio_clips(wav_files, out_dir, speaker_id):
    # Only use original high-quality .wav files (not *_16k.wav)
    wavs = sorted([w for w in wav_files if w.exists() and not w.name.endswith('_16k.wav')], key=lambda x: x.name)
    if not wavs:
        print(f"[WARN] No high-quality .wav files found for {speaker_id}, skipping audio sample generation.")
        return []
    else:
        print(f"[AUDIO SAMPLE] Using files for {speaker_id}: {[w.name for w in wavs]}")
    arrays = []
    sample_rate = None
    total_samples = 0
    for wav in wavs:
        try:
            data, sr = sf.read(str(wav))
            print(f"[AUDIO SAMPLE] {wav.name}: sr={sr}, shape={data.shape}, dtype={data.dtype}")
            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                print(f"[ERROR] Sample rate mismatch: {wav.name} is {sr} Hz, expected {sample_rate} Hz. Skipping.")
                continue
            arrays.append(data)
            total_samples += data.shape[0]
            if total_samples / sample_rate >= 30:
                break
        except Exception as e:
            print(f"[ERROR] Could not read {wav.name}: {e}")
            continue
    if not arrays:
        print(f"[WARN] No valid segments found for {speaker_id} after filtering.")
        return []
    combined = np.concatenate(arrays, axis=0)
    outputs = []
    for sec in [15, 30]:
        num_samples = int(sec * sample_rate)
        clip = combined[:num_samples]
        if clip.shape[0] == 0:
            continue
        out_wav = out_dir / f"{speaker_id}_{sec}s.wav"
        sf.write(str(out_wav), clip, sample_rate)
        print(f"[AUDIO SAMPLE] {out_wav}: sr={sample_rate}, shape={clip.shape}, dtype={clip.dtype}, size={out_wav.stat().st_size} bytes")
        outputs.append(str(out_wav))
        # Optional: convert to MP3 using ffmpeg (if needed)
        try:
            import subprocess
            out_mp3 = out_dir / f"{speaker_id}_{sec}s.mp3"
            cmd = [
                "ffmpeg", "-y", "-i", str(out_wav),
                "-ar", str(sample_rate), "-acodec", "libmp3lame", "-b:a", "320k", str(out_mp3)
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"[AUDIO SAMPLE] {out_mp3}: (converted from WAV)")
            outputs.append(str(out_mp3))
        except Exception as e:
            print(f"[AUDIO SAMPLE] Could not convert to MP3: {e}")
    if combined.shape[0] < 15000:
        print(f"[WARN] Only {combined.shape[0]/sample_rate:.2f}s audio available for {speaker_id}, samples may be short.")
    return outputs

class CharacterPersonaBuilder(ExtensionBase):
    def __init__(self, output_root, llm_config_path=None, max_tokens=4096):
        super().__init__(output_root)
        self.llm_config_path = llm_config_path or 'workflows/llm_tasks.json'
        self.llm_config = self._load_llm_config()
        self.max_tokens = max_tokens

    def _load_llm_config(self):
        config_path = Path(self.llm_config_path)
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            if 'lm_studio_base_url' in config:
                return config
            elif 'llm_tasks' in config and 'llm_config' in config:
                return config['llm_config']
        return {
            'lm_studio_base_url': 'http://localhost:1234/v1',
            'lm_studio_api_key': 'lm-studio',
            'lm_studio_model_identifier': 'llama-3.1-8b-supernova-etherealhermes',
            'lm_studio_temperature': 0.5,
            'lm_studio_max_tokens': 2048
        }

    def get_call_title(self, call_id):
        llm_dir = self.output_root / 'llm' / call_id
        title_file = llm_dir / 'call_title.txt'
        if title_file.exists():
            title = title_file.read_text(encoding='utf-8').strip()
            sanitized = sanitize_title(title)
            if sanitized:
                return sanitized
        return call_id

    def build_user_prompt(self, transcript, call_synopsis=None):
        lines = [l for l in transcript.strip().splitlines() if l.strip()]
        if len(lines) > 5:
            prompt = '[SPEAKER TRANSCRIPT]\n' + transcript.strip()
            self.log(f"[PROMPT] Using transcript only ({len(lines)} lines), omitting call_synopsis.")
        else:
            prompt = '[SPEAKER TRANSCRIPT]\n' + transcript.strip()
            if call_synopsis:
                prompt += '\n\n[CALL SYNOPSIS]\n' + call_synopsis.strip()
                self.log(f"[PROMPT] Using transcript + call_synopsis (only {len(lines)} lines).")
            else:
                self.log(f"[PROMPT] Using transcript only (≤5 lines, no call_synopsis available).")
        return prompt

    def generate_persona(self, system_prompt, user_prompt, out_dir=None):
        # Log and save both system and user prompts
        self.log(f"[SYSTEM PROMPT] (truncated for display) {system_prompt[:500]}... (truncated)")
        self.log(f"[USER PROMPT] (truncated for display) {user_prompt[:500]}... (truncated)")
        if out_dir is not None:
            (out_dir / 'persona_system_prompt.txt').write_text(system_prompt, encoding='utf-8')
            (out_dir / 'persona_user_prompt.txt').write_text(user_prompt, encoding='utf-8')
        # Call LLM with system prompt and user prompt
        # Here we assume run_llm_task supports system/user separation; if not, concatenate as fallback
        try:
            response = run_llm_task(user_prompt, self.llm_config, system_prompt=system_prompt, single_output=True, chunking=False)
        except TypeError:
            # Fallback: concatenate system and user prompt if system_prompt arg not supported
            response = run_llm_task(system_prompt + '\n\n' + user_prompt, self.llm_config, single_output=True, chunking=False)
        persona = response.strip()
        self.log(f"[LLM RESPONSE] (truncated for display) {persona[:500]}... (truncated)")
        if out_dir is not None:
            (out_dir / 'persona_llm_response.txt').write_text(persona, encoding='utf-8')
        return persona

    def run(self):
        speakers_root = self.output_root / 'speakers'
        characters_root = self.output_root / 'characters'
        characters_root.mkdir(exist_ok=True)
        llm_root = self.output_root / 'llm'
        if not speakers_root.exists():
            self.log("No speakers directory found.")
            return
        persona_manifest = []
        for call_folder in speakers_root.iterdir():
            if not call_folder.is_dir():
                continue
            call_id = call_folder.name
            call_title = self.get_call_title(call_id)
            for channel_folder in call_folder.iterdir():
                if not channel_folder.is_dir():
                    continue
                channel = channel_folder.name
                for speaker_folder in channel_folder.iterdir():
                    if not speaker_folder.is_dir() or not speaker_folder.name.startswith('S'):
                        continue
                    speaker_id = speaker_folder.name
                    transcript_path = speaker_folder / 'speaker_transcript.txt'
                    if not transcript_path.exists():
                        self.log(f"[SKIP] {transcript_path} does not exist, skipping.")
                        continue
                    transcript_bytes = transcript_path.stat().st_size
                    if transcript_bytes < 300:
                        self.log(f"[SKIP] {transcript_path} is only {transcript_bytes} bytes (<300), skipping persona generation.")
                        continue
                    transcript = transcript_path.read_text(encoding='utf-8').strip()
                    # Sum durations of all .wav files for this speaker
                    wav_files = list(speaker_folder.glob('*.wav'))
                    total_duration = 0.0
                    for wav_file in wav_files:
                        try:
                            info = mediainfo(str(wav_file))
                            duration = float(info.get('duration', 0))
                            total_duration += duration
                        except Exception:
                            pass
                    if total_duration < 15.0:
                        self.log(f"[SKIP] {speaker_folder} has only {total_duration:.2f}s audio (<15s), skipping persona generation.")
                        continue
                    # Compose user prompt: transcript + call synopsis (if available)
                    call_synopsis_path = llm_root / call_id / 'call_synopsis.txt'
                    call_synopsis = call_synopsis_path.read_text(encoding='utf-8').strip() if call_synopsis_path.exists() else None
                    user_prompt = self.build_user_prompt(transcript, call_synopsis)
                    out_dir = characters_root / call_id / channel / speaker_id
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_transcript_path = out_dir / 'speaker_transcript.txt'
                    out_transcript_path.write_text(transcript, encoding='utf-8')
                    persona_text = self.generate_persona(CHARACTER_AI_GUIDELINES, user_prompt, out_dir=out_dir)
                    persona_path = out_dir / 'persona.md'
                    persona_path.write_text(persona_text, encoding='utf-8')
                    # Create audio samples (15s, 30s)
                    audio_samples = create_audio_clips(wav_files, out_dir, speaker_id)
                    if audio_samples:
                        self.log(f"[OK] Created audio samples for {speaker_id} in {call_id}/{channel}: {audio_samples}")
                    else:
                        self.log(f"[WARN] No audio samples created for {speaker_id} in {call_id}/{channel}")
                    persona_manifest.append({
                        'call_id': call_id,
                        'channel': channel,
                        'speaker': speaker_id,
                        'persona_path': str(persona_path),
                        'source_audio_paths': [str(w) for w in wav_files if w.exists()],
                        'audio_samples': audio_samples
                    })
                    self.log(f"[OK] Persona, transcript for {speaker_id} in {call_id}/{channel} written to {out_dir}")
        # Write persona manifest
        manifest_path = characters_root / 'persona_manifest.json'
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(persona_manifest, f, indent=2)
        self.log(f"Persona manifest written to {manifest_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Character Persona Builder Extension (refactored)")
    parser.add_argument('output_root', type=str, help='Root output folder (parent of speakers/)')
    parser.add_argument('--llm-config', type=str, default=None, help='Path to LLM config JSON (default: workflows/llm_tasks.json)')
    parser.add_argument('--max-tokens', type=int, default=4096, help='Max tokens per LLM chunk (default: 4096, max: 8192).')
    args = parser.parse_args()
    max_tokens = min(args.max_tokens, 8192)
    ext = CharacterPersonaBuilder(args.output_root, llm_config_path=args.llm_config, max_tokens=max_tokens)
    ext.run() 