import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Callable, Dict, Any
from tqdm import tqdm
from rich.console import Console
from rich.traceback import install as rich_traceback_install
import uuid
import numpy as np
from file_ingestion import process_file_job, SUPPORTED_EXTENSIONS, TYPE_MAP
from audio_separation import separate_audio_file, separate_single_audio_file
from clap_annotation import annotate_clap_for_out_files, segment_audio_with_clap
from speaker_diarization import batch_diarize, segment_speakers_from_diarization
from transcription import transcribe_segments
import torchaudio
import re
import soundfile as sf
from collections import defaultdict
# --- Finalization stage import ---
from finalization_stage import run_finalization_stage
# --- Resume functionality imports ---
from resume_utils import add_resume_to_orchestrator, run_stage_with_resume, should_resume_pipeline, print_resume_status, print_stage_status
import random
import hashlib
import yaml
import tempfile
import shutil
import subprocess
import mutagen

rich_traceback_install()
console = Console()

# Sub-identifier mapping for tuple members
TUPLE_SUBID = {
    'left': 'a',      # recv_out
    'right': 'b',     # trans_out
    'out': 'c',       # out
}

# Supported video extensions for audio extraction
VIDEO_EXTENSIONS = {'.mkv', '.mp4', '.avi', '.mov', '.webm'}

class Job:
    def __init__(self, job_id: str, data: Dict[str, Any], job_type: str = 'rename'):
        self.job_id = job_id
        self.data = data  # Arbitrary metadata/state for the job
        self.state = {}
        self.success = True
        self.error = None
        self.progress = 0.0
        self.job_type = job_type  # 'rename' or 'separate'

class PipelineOrchestrator:
    def __init__(self, run_folder: Path, asr_engine: str, llm_config_path: str = None):
        self.jobs: List[Job] = []
        self.log: List[Dict[str, Any]] = []
        self.manifest: List[Dict[str, Any]] = []
        self.run_folder = run_folder
        self._log_buffer: List[Dict[str, Any]] = []  # Buffer for log events
        self._console_buffer: List[str] = []         # Buffer for console output
        self._logging_enabled = False                # Only enable after PII is gone
        self.asr_engine = asr_engine
        self.llm_config_path = llm_config_path
        self.global_llm_seed = None
    def add_job(self, job: Job):
        self.jobs.append(job)
    def log_event(self, level, event, details=None):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'event': event,
            'details': details or {}
        }
        self.log.append(entry)
        # Buffer console output until logging is enabled (after PII is gone)
        msg = None
        if level == 'ERROR':
            msg = f"[bold red]{level}[/] {event}: {details}"
        elif level == 'WARNING':
            msg = f"[yellow]{level}[/] {event}: {details}"
        else:
            msg = f"[green]{level}[/] {event}: {details}"
        if self._logging_enabled:
            console.print(msg)
        else:
            self._console_buffer.append(msg)
    def enable_logging(self):
        """
        Call this after all raw_inputs are deleted and outputs are anonymized.
        Flushes buffered logs to console.
        """
        self._logging_enabled = True
        for msg in self._console_buffer:
            console.print(msg)
        self._console_buffer.clear()
    def write_log(self):
        # Ensure run_folder exists before writing
        self.run_folder.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        log_path = self.run_folder / f'orchestrator-log-{ts}.json'
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.log, f, indent=2)
        if self._logging_enabled:
            console.print(f"[bold green]Orchestrator log written to {log_path}[/]")
        else:
            self._console_buffer.append(f"[bold green]Orchestrator log written to {log_path}[/]")
    def write_manifest(self):
        # Ensure run_folder exists before writing  
        self.run_folder.mkdir(parents=True, exist_ok=True)
        manifest_path = self.run_folder / 'manifest.json'
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(self.manifest, f, indent=2)
        if self._logging_enabled:
            console.print(f"[bold green]Manifest written to {manifest_path}[/]")
        else:
            self._console_buffer.append(f"[bold green]Manifest written to {manifest_path}[/]")
    def add_separation_jobs(self):
        """
        After renaming, create jobs for audio separation for each file in renamed/.
        Only single files are supported (no tuple/call or left/right logic).
        """
        renamed_dir = self.run_folder / 'renamed'
        if not renamed_dir.exists():
            self.log_event('WARNING', 'renamed_directory_missing', {
                'renamed_dir': str(renamed_dir),
                'message': 'No renamed directory found, skipping separation job creation'
            })
            return
        renamed_files = list(renamed_dir.glob('*'))
        if not renamed_files:
            self.log_event('WARNING', 'renamed_directory_empty', {
                'renamed_dir': str(renamed_dir),
                'message': 'Renamed directory is empty, no separation jobs to create'
            })
            return
        separation_job_count = 0
        for file in renamed_dir.iterdir():
            if file.is_file():
                # Skip files under 200KB
                if file.stat().st_size < 200 * 1024:
                    self.log_event('WARNING', 'file_too_small_for_separation', {'file': str(file), 'size_bytes': file.stat().st_size})
                    continue
                input_path = file
                if input_path.suffix.lower() != '.wav':
                    import soundfile as sf
                    audio, sr = sf.read(str(input_path))
                    wav_path = input_path.with_suffix('.wav')
                    sf.write(str(wav_path), audio, sr)
                    self.log_event('INFO', 'converted_to_wav', {'original': str(input_path), 'converted': str(wav_path)})
                    input_path = wav_path
                # Extract metadata using mutagen
                import mutagen
                metadata = {}
                try:
                    audio_file = mutagen.File(str(file), easy=True)
                    if audio_file is not None:
                        for key, value in audio_file.items():
                            if key.lower() not in ['filename', 'file', 'path']:
                                metadata[key] = value
                    self.log_event('INFO', 'metadata_extracted', {'input': str(file), 'metadata': metadata})
                except Exception as e:
                    self.log_event('WARNING', 'metadata_extraction_failed', {'input': str(file), 'error': str(e)})
                separated_dir = self.run_folder / 'separated' / f"{separation_job_count:04d}"
                separated_dir.mkdir(parents=True, exist_ok=True)
                metadata_path = separated_dir / 'metadata.json'
                try:
                    import json
                    with open(metadata_path, 'w', encoding='utf-8') as mf:
                        json.dump(metadata, mf, indent=2)
                    self.log_event('INFO', 'metadata_written', {'output': str(metadata_path)})
                except Exception as e:
                    self.log_event('WARNING', 'metadata_write_failed', {'output': str(metadata_path), 'error': str(e)})
                job_data = {
                    'input_path': str(input_path),
                    'input_name': input_path.name,
                    'separated_dir': str(separated_dir),
                    'metadata_path': str(metadata_path)
                }
                job_id = f"separate_{separation_job_count:04d}_singlefile"
                self.jobs.append(Job(job_id=job_id, data=job_data, job_type='separate'))
                separation_job_count += 1
                self.log_event('INFO', 'separation_job_created_single', {'input': str(input_path), 'separated_dir': str(separated_dir)})
        self.log_event('INFO', 'separation_jobs_created', {
            'separation_job_count': separation_job_count,
            'total_files': len(renamed_files)
        })

    def run_audio_separation_stage(self):
        """
        Run audio separation for all jobs of type 'separate'.
        Updates manifest and logs via orchestrator methods.
        After separation, move/copy all output stems (vocals.wav, instrumental.wav, etc.) from the nested subfolder up to the separated/<index>/ directory, renaming as <index>-<stem_type>.wav. Update manifest to reference the new paths. Remove the now-empty subfolder if possible.
        """
        import json
        separated_dir = self.run_folder / 'separated'
        separated_dir.mkdir(exist_ok=True)
        model_path = 'mel_band_roformer_vocals_fv4_gabox.ckpt'
        separation_jobs = [job for job in self.jobs if job.job_type == 'separate']
        self.log_event('INFO', 'audio_separation_start', {'file_count': len(separation_jobs)})
        for job in separation_jobs:
            input_file = Path(job.data['input_path'])
            job_separated_dir = Path(job.data.get('separated_dir')) if 'separated_dir' in job.data else separated_dir
            self.log_event('INFO', 'separation_model_invocation', {'input_file': str(input_file), 'output_dir': str(job_separated_dir)})
            result = separate_audio_file(input_file, job_separated_dir, model_path)
            if result['separation_status'] == 'success':
                self.log_event('INFO', 'audio_separation_success', {
                    'input_name': result['input_name'],
                    'output_stems': [s['output_path'] for s in result['output_stems']]
                })
                job.success = True
                # Move/copy stems up one level and rename
                for s in result['output_stems']:
                    orig_path = Path(s['output_path'])
                    stem_type = s.get('stem_type', orig_path.stem)
                    new_name = f"{job_separated_dir.name}-{stem_type}.wav"
                    new_path = job_separated_dir / new_name
                    if orig_path.exists() and orig_path != new_path:
                        import shutil
                        shutil.copy2(orig_path, new_path)
                        self.log_event('INFO', 'stem_moved', {'from': str(orig_path), 'to': str(new_path)})
                # Optionally remove the now-empty subfolder
                for s in result['output_stems']:
                    orig_path = Path(s['output_path'])
                    if orig_path.parent != job_separated_dir:
                        try:
                            import os
                            if orig_path.exists():
                                os.remove(orig_path)
                            if not any(orig_path.parent.iterdir()):
                                os.rmdir(orig_path.parent)
                        except Exception as e:
                            self.log_event('WARNING', 'stem_cleanup_failed', {'folder': str(orig_path.parent), 'error': str(e)})
                # Update manifest to reference new paths
                updated_stems = []
                for s in result['output_stems']:
                    stem_type = s.get('stem_type', Path(s['output_path']).stem)
                    new_name = f"{job_separated_dir.name}-{stem_type}.wav"
                    new_path = job_separated_dir / new_name
                    updated_stems.append({'stem_type': stem_type, 'output_path': str(new_path)})
                if 'metadata_path' in job.data:
                    vocals_stem = None
                    for s in updated_stems:
                        if 'vocals' in s['stem_type']:
                            vocals_stem = s['output_path']
                            break
                    if vocals_stem:
                        import mutagen
                        try:
                            with open(job.data['metadata_path'], 'r', encoding='utf-8') as mf:
                                metadata = json.load(mf)
                            audio = mutagen.File(vocals_stem, easy=True)
                            if audio is not None:
                                for key, value in metadata.items():
                                    audio[key] = value
                                audio.save()
                            self.log_event('INFO', 'metadata_propagated', {'output': vocals_stem, 'metadata': metadata})
                        except Exception as e:
                            self.log_event('WARNING', 'metadata_propagation_failed', {'output': vocals_stem, 'error': str(e)})
                self.manifest.append({
                    'stage': 'separated',
                    'input_name': result['input_name'],
                    'output_stems': updated_stems,
                    'separation_status': result['separation_status']
                })
            else:
                self.log_event('ERROR', 'audio_separation_failed', {
                    'input_name': result['input_name'],
                    'stderr': result['stderr']
                })
                job.success = False
                job.error = result['stderr']
        manifest_path = separated_dir / 'separation_manifest.json'
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump([{
                'input_name': job.data['input_name'],
                'success': job.success,
                'error': job.error
            } for job in separation_jobs], f, indent=2)
        self.log_event('INFO', 'audio_separation_complete', {'manifest_path': str(manifest_path)})

    def run_diarization_stage(self, hf_token=None, min_speakers=None):
        """
        Run speaker diarization for all vocal files in separated/<index>/.
        Only single files are supported (no tuple/call or left/right logic).
        Updates manifest and logs via orchestrator methods.
        """
        separated_dir = self.run_folder / 'separated'
        diarized_dir = self.run_folder / 'diarized'
        diarized_dir.mkdir(exist_ok=True)
        self.log_event('INFO', 'diarization_start', {'max_speakers': 8})
        audio_files = []
        for sep_subdir in separated_dir.iterdir():
            if not sep_subdir.is_dir():
                continue
            for wav_file in sep_subdir.glob('*.wav'):
                audio_files.append(wav_file)
        if not audio_files:
            self.log_event('WARNING', 'no_audio_files_found', {
                'separated_dir': str(separated_dir),
                'message': 'No audio files found for diarization'
            })
        results = batch_diarize(
            str(separated_dir),
            str(diarized_dir),
            hf_token=hf_token,
            min_speakers=min_speakers,
            max_speakers=8,
            progress=True
        )
        for result in results:
            self.log_and_manifest(
                stage='diarized',
                call_id=None,
                input_files=[str(result.get('input_name'))],
                output_files=[str(result.get('json'))],
                params={'max_speakers': 8},
                metadata={'segments': len(result.get('segments', []))},
                event='diarization_result',
                result='success'
            )
            self.manifest.append({
                'stage': 'diarized',
                'call_id': result.get('call_id'),
                'input_name': result.get('input_name'),
                'rttm': result.get('rttm'),
                'json': result.get('json'),
                'segments': result.get('segments', [])
            })
        self.log_event('INFO', 'diarization_complete', {'count': len(results), 'max_speakers': 8})

    def run_speaker_segmentation_stage(self):
        """
        Segment each diarized *-vocals.wav file into per-speaker audio files, saving in speakers/<call id>/SXX/.
        Log segment metadata and update manifest.
        Logs every file written and updates manifest.
        """
        diarized_dir = self.run_folder / 'diarized'
        separated_dir = self.run_folder / 'separated'
        speakers_dir = self.run_folder / 'speakers'
        speakers_dir.mkdir(exist_ok=True)
        self.log_event('INFO', 'speaker_segmentation_start', {})
        results = segment_speakers_from_diarization(
            str(diarized_dir),
            str(separated_dir),
            str(speakers_dir),
            progress=True
        )
        for seg in results:
            self.log_and_manifest(
                stage='speaker_segmented',
                call_id=seg.get('call_id'),
                input_files=[str(seg.get('input_wav'))],
                output_files=[str(seg.get('wav'))],
                params={'channel': seg.get('channel'), 'speaker': seg.get('speaker')},
                metadata={'start': seg.get('start'), 'end': seg.get('end')},
                event='speaker_segment',
                result='success'
            )
            self.manifest.append({
                'stage': 'speaker_segmented',
                **seg
            })
        self.log_event('INFO', 'speaker_segmentation_complete', {'count': len(results)})

    def run_resample_segments_stage(self):
        """
        Resample all speaker segment WAVs to 16kHz mono for ASR. Save as <segment>_16k.wav and update manifest.
        Overwrites any old _16k.wav files. Adds debug logging for waveform shape.
        Logs every file written and updates manifest.
        Skips and logs empty or unreadable files.
        """
        speakers_dir = self.run_folder / 'speakers'
        resampled_segments = []
        self.log_event('INFO', 'resample_segments_start', {})
        for entry in self.manifest:
            if entry.get('stage') == 'speaker_segmented' and 'wav' in entry:
                wav_path = Path(entry['wav'])
                if not wav_path.exists():
                    continue
                out_path = wav_path.with_name(wav_path.stem + '_16k.wav')
                if out_path.exists():
                    out_path.unlink()
                try:
                    waveform, sr = torchaudio.load(str(wav_path))
                    print(f"[DEBUG] Loaded: {wav_path} shape={waveform.shape} sr={sr} dtype={waveform.dtype}")
                    if waveform.numel() == 0 or (waveform.ndim == 2 and waveform.shape[1] == 0):
                        warn_msg = f"Segment file is empty, skipping: {wav_path.name}"
                        print(f"[WARN] {warn_msg}")
                        self.log_and_manifest(
                            stage='resampled_skipped',
                            call_id=entry.get('call_id'),
                            input_files=[str(wav_path)],
                            output_files=[],
                            params={'reason': 'empty file'},
                            metadata=None,
                            event='file_skipped',
                            result='skipped',
                            error=warn_msg
                        )
                        continue
                    if waveform.ndim == 3:
                        if waveform.shape[0] == 1 and waveform.shape[2] == 2:
                            waveform = waveform.squeeze(0).mean(dim=1, keepdim=True).transpose(0, 1)
                            print(f"[DEBUG] Squeezed and averaged: shape={waveform.shape}")
                        elif waveform.shape[0] == 1 and waveform.shape[1] == 2:
                            waveform = waveform.squeeze(0).mean(dim=0, keepdim=True)
                            print(f"[DEBUG] Squeezed and averaged: shape={waveform.shape}")
                        else:
                            print(f"[WARN] Unexpected 3D shape: {waveform.shape}")
                    elif waveform.ndim == 2:
                        if waveform.shape[0] > 1:
                            waveform = waveform.mean(dim=0, keepdim=True)
                            print(f"[DEBUG] Averaged channels: shape={waveform.shape}")
                    else:
                        print(f"[WARN] Unexpected waveform ndim: {waveform.ndim} shape={waveform.shape}")
                    if sr != 16000:
                        waveform = torchaudio.functional.resample(waveform, sr, 16000)
                        sr = 16000
                        print(f"[DEBUG] After resample: shape={waveform.shape} sr={sr}")
                    if waveform.ndim != 2 or waveform.shape[0] != 1:
                        print(f"[WARN] Skipping {wav_path}: shape after mono/resample is {waveform.shape}, expected [1, time]")
                        continue
                    torchaudio.save(str(out_path), waveform, sr)
                    self.log_and_manifest(
                        stage='resampled',
                        call_id=entry.get('call_id'),
                        input_files=[str(wav_path)],
                        output_files=[str(out_path)],
                        params={'target_sr': 16000},
                        metadata=None,
                        event='file_written',
                        result='success'
                    )
                    resampled_entry = dict(entry)
                    resampled_entry['wav_16k'] = str(out_path)
                    resampled_entry['stage'] = 'resampled'
                    self.manifest.append(resampled_entry)
                    resampled_segments.append(resampled_entry)
                except Exception as e:
                    warn_msg = f"Resample failed for {wav_path.name}: {str(e)}"
                    print(f"[WARN] {warn_msg}")
                    self.log_and_manifest(
                        stage='resampled_skipped',
                        call_id=entry.get('call_id'),
                        input_files=[str(wav_path)],
                        output_files=[],
                        params={'reason': 'exception'},
                        metadata=None,
                        event='file_skipped',
                        result='skipped',
                        error=warn_msg
                    )
                    continue
        self.log_event('INFO', 'resample_segments_complete', {'count': len(resampled_segments)})

    def run_transcription_stage(self, asr_engine='parakeet', asr_config=None):
        """
        Transcribe all speaker segments using the selected ASR engine (parakeet or whisper).
        Updates manifest with transcript results.
        Logs every file written and updates manifest.
        Tracks and logs failed/skipped transcriptions for full auditability.
        """
        speakers_dir = self.run_folder / 'speakers'
        segments = [entry for entry in self.manifest if entry.get('stage') == 'resampled' and 'wav_16k' in entry]
        if not segments:
            segments = []
            for call_id in os.listdir(speakers_dir):
                call_dir = speakers_dir / call_id
                if not call_dir.is_dir():
                    continue
                for channel in os.listdir(call_dir):
                    channel_dir = call_dir / channel
                    if not channel_dir.is_dir():
                        continue
                    for spk in os.listdir(channel_dir):
                        spk_dir = channel_dir / spk
                        if not spk_dir.is_dir():
                            continue
                        for wav in os.listdir(spk_dir):
                            if wav.endswith('_16k.wav'):
                                wav_path = spk_dir / wav
                                parts = wav.replace('_16k','')[:-4].split('-')
                                index = int(parts[0]) if parts and parts[0].isdigit() else None
                                start = float(parts[1])/100 if len(parts) > 1 else None
                                end = float(parts[2])/100 if len(parts) > 2 else None
                                segments.append({
                                    'call_id': call_id,
                                    'channel': channel,
                                    'speaker': spk,
                                    'index': index,
                                    'start': start,
                                    'end': end,
                                    'wav': str(wav_path)
                                })
        if asr_config is None:
            asr_config = {}
        asr_config = {**asr_config, 'asr_engine': asr_engine}
        results = transcribe_segments(segments, asr_config)
        success_count = 0
        fail_count = 0
        skip_count = 0
        for seg, res in zip(segments, results):
            if res.get('error'):
                self.log_and_manifest(
                    stage='transcription_failed',
                    call_id=seg.get('call_id'),
                    input_files=[str(seg.get('wav'))],
                    output_files=[],
                    params={'asr_engine': asr_engine},
                    metadata={'error': res.get('error')},
                    event='transcription',
                    result='error',
                    error=res.get('error')
                )
                fail_count += 1
                continue
            # Check for missing .txt or .json
            if not res.get('txt') or not os.path.exists(res.get('txt')):
                warn_msg = f"Transcription .txt missing for {seg.get('wav')}"
                self.log_and_manifest(
                    stage='transcription_skipped',
                    call_id=seg.get('call_id'),
                    input_files=[str(seg.get('wav'))],
                    output_files=[],
                    params={'asr_engine': asr_engine},
                    metadata={'warning': warn_msg},
                    event='transcription',
                    result='skipped',
                    error=warn_msg
                )
                skip_count += 1
                continue
            self.log_and_manifest(
                stage='transcribed',
                call_id=res.get('call_id'),
                input_files=[str(res.get('wav'))],
                output_files=[str(res.get('txt')), str(res.get('json'))] if res.get('txt') and res.get('json') else [],
                params={'asr_engine': asr_engine},
                metadata={'error': res.get('error') if 'error' in res else None},
                event='transcription',
                result='success',
                error=res.get('error') if 'error' in res else None
            )
            self.manifest.append({
                **res,
                'stage': 'transcribed'
            })
            success_count += 1
        self.log_event('INFO', 'transcription_complete', {'count': len(results), 'success': success_count, 'failed': fail_count, 'skipped': skip_count})
        print(f"[SUMMARY] Transcription: {success_count} succeeded, {fail_count} failed, {skip_count} skipped.")

    @staticmethod
    def sanitize(text, max_words=12, max_length=40):
        if not text:
            return 'untitled'
        words = re.findall(r'\w+', text)[:max_words]
        short = '_'.join(words)
        return short[:max_length] or 'untitled'

    def run_rename_soundbites_stage(self):
        """
        No longer renames files in speakers/ folder. Renaming and copying now happens in run_final_soundbite_stage.
        This stage is now a no-op for file renaming, but still updates manifest for valid soundbites.
        Skips soundbites with missing/empty transcript or duration < 1s.
        Logs every file written and updates manifest.
        """
        valid_count = 0
        for entry in self.manifest:
            if entry.get('stage') == 'transcribed' and 'wav' in entry and 'text' in entry:
                transcript = entry['text']
                if not transcript or not transcript.strip():
                    continue
                wav_path = Path(entry['wav'])
                try:
                    info = sf.info(str(wav_path))
                    duration = info.duration
                except Exception:
                    duration = None
                if duration is not None and duration < 1.0:
                    continue
                entry['stage'] = 'soundbite_valid'
                valid_count += 1
                self.log_and_manifest(
                    stage='soundbite_valid',
                    call_id=entry.get('call_id'),
                    input_files=[str(entry.get('wav'))],
                    output_files=[str(entry.get('wav'))],
                    params=None,
                    metadata={'duration': duration},
                    event='soundbite_valid',
                    result='success'
                )
        self.log_event('INFO', 'soundbite_valid_count', {'count': valid_count})

    def run_final_soundbite_stage(self):
        """
        Copy and rename valid soundbites from speakers/ to soundbites/ folder, using <index>-<short_transcription>.* naming.
        Only includes valid soundbites (with transcript and duration >= 1s).
        Formats master transcript as [SpeakerXX][start-end]: Transcription.
        Also generates new segment logs in soundbites/<segment_index>/<speaker>/ with transcript text.
        The master transcript is the canonical input for LLM tasks.
        Integrates CLAP events with confidence >= 0.90 as [CLAP][start-end]: <label> (high confidence), sorted chronologically.
        Logs every file written and updates manifest.
        Filters out malformed segment entries.
        """
        import shutil
        import json
        from collections import defaultdict
        speakers_dir = self.run_folder / 'speakers'
        soundbites_dir = self.run_folder / 'soundbites'
        soundbites_dir.mkdir(exist_ok=True)
        segments = [entry for entry in self.manifest if entry.get('stage') == 'speaker_segmented']
        # Process each segment independently
        master_soundbites = []
        for seg in segments:
            idx = seg.get('index')
            channel = seg.get('channel')
            speaker = seg.get('speaker')
            start = seg.get('start')
            end = seg.get('end')
            duration = end - start if end is not None and start is not None else 0
            if duration < 1.0:
                continue
            if not channel:
                self.log_event('WARNING', 'channel_missing', {'idx': idx, 'speaker': speaker})
                continue
            # Use call_id for the top-level directory, not idx
            call_id = seg.get('call_id', '0000')
            spk_dir = speakers_dir / call_id / channel / speaker
            if not spk_dir.exists():
                self.log_event('WARNING', 'speaker_dir_missing', {'spk_dir': str(spk_dir), 'idx': idx, 'channel': channel, 'speaker': speaker})
                continue
            # Use the full unique base filename for this segment
            wav_path = seg.get('wav')
            if not wav_path:
                self.log_event('WARNING', 'wav_path_missing', {'seg': seg})
                continue
            base_name = Path(wav_path).stem
            orig_wav = spk_dir / (base_name + '.wav')
            orig_txt = spk_dir / (base_name + '.txt')
            orig_json = spk_dir / (base_name + '.json')
            out_dir = soundbites_dir / f"{idx:04d}" / channel / speaker
            out_dir.mkdir(parents=True, exist_ok=True)
            # --- FIX: transcript lookup and check ---
            transcript_entry = next((e for e in self.manifest if e.get('stage') == 'soundbite_valid' and e.get('index') == idx and e.get('speaker') == speaker and e.get('channel') == channel), None)
            transcript = transcript_entry.get('text') if transcript_entry and transcript_entry.get('text') else None
            if not transcript or not transcript.strip():
                continue
            # --- END FIX ---
            out_base = f"{idx:04d}-{self.sanitize(transcript)}"
            out_wav = out_dir / (out_base + '.wav')
            out_txt = out_dir / (out_base + '.txt') if orig_txt.exists() else None
            out_json = out_dir / (out_base + '.json') if orig_json.exists() else None
            if orig_wav.exists():
                shutil.copy2(orig_wav, out_wav)
            if orig_txt and orig_txt.exists() and out_txt:
                shutil.copy2(orig_txt, out_txt)
            if orig_json and orig_json.exists() and out_json:
                shutil.copy2(orig_json, out_json)
            self.log_and_manifest(
                stage='final_soundbite',
                call_id=None,
                input_files=[str(orig_wav), str(orig_txt) if orig_txt else None, str(orig_json) if orig_json else None],
                output_files=[str(out_wav), str(out_txt) if out_txt else None, str(out_json) if out_json else None],
                params={'speaker': speaker, 'channel': channel},
                metadata={'duration': duration},
                event='file_written',
                result='success'
            )
            manifest_entry = dict(seg)
            manifest_entry['stage'] = 'final_soundbite'
            manifest_entry['soundbite_wav'] = str(out_wav)
            manifest_entry['transcript'] = transcript
            manifest_entry['txt'] = str(out_txt) if out_txt else None
            manifest_entry['json'] = str(out_json) if out_json else None
            self.manifest.append(manifest_entry)
            master_soundbites.append({
                'spk_fmt': f"[Speaker{speaker}]",
                'start': start,
                'end': end,
                'transcript': transcript
            })
        # CLAP events (optional, if present)
        clap_events = []
        clap_dir = self.run_folder / 'clap'
        if clap_dir.exists():
            for clap_json in clap_dir.glob('**/*_clap_annotations.json'):
                with open(clap_json, 'r', encoding='utf-8') as f:
                    clap_data = json.load(f)
                for event in clap_data.get('events', []):
                    if event.get('confidence', 0) >= 0.90:
                        clap_events.append({
                            'start': event.get('start', 0),
                            'end': event.get('end', 0),
                            'label': event.get('label', 'unknown'),
                            'confidence': event.get('confidence')
                        })
        # Compose master transcript
        all_events = [
            {
                'type': 'soundbite',
                'start': s['start'],
                'end': s['end'],
                'text': f"{s['spk_fmt']}[{s['start']:.2f}-{s['end']:.2f}]: {s['transcript']}"
            } for s in master_soundbites
        ] + [
            {
                'type': 'clap',
                'start': e['start'],
                'end': e['end'],
                'text': f"[CLAP][{e['start']:.2f}-{e['end']:.2f}]: {e['label']} (high confidence)"
            } for e in clap_events
        ]
        all_events_sorted = sorted(all_events, key=lambda x: x['start'])
        master_txt = soundbites_dir / 'master_transcript.txt'
        if not master_soundbites:
            self.log_event('WARNING', 'no_valid_soundbites', {})
            return
        with open(master_txt, 'w', encoding='utf-8') as f:
            for ev in all_events_sorted:
                f.write(f"{ev['text']}\n")
        self.log_and_manifest(
            stage='master_transcript',
            call_id=None,
            input_files=None,
            output_files=[str(master_txt)],
            params=None,
            metadata=None,
            event='file_written',
            result='success'
        )
        master_json = soundbites_dir / 'master_transcript.json'
        with open(master_json, 'w', encoding='utf-8') as f:
            json.dump(all_events_sorted, f, indent=2, ensure_ascii=False)
        self.log_and_manifest(
            stage='master_transcript',
            call_id=None,
            input_files=None,
            output_files=[str(master_json)],
            params=None,
            metadata=None,
            event='file_written',
            result='success'
        )
        # Write per-speaker transcripts
        per_speaker_transcripts = defaultdict(list)
        for s in master_soundbites:
            speaker = s['spk_fmt']
            start = s['start']
            end = s['end']
            transcript = s['transcript']
            if start is not None and end is not None:
                line = f"[{start:.2f}-{end:.2f}] {transcript}"
            else:
                line = transcript
            per_speaker_transcripts[speaker].append(line)
        speakers_dir_out = soundbites_dir / 'per_speaker_transcripts'
        speakers_dir_out.mkdir(parents=True, exist_ok=True)
        for speaker, utterances in per_speaker_transcripts.items():
            transcript_path = speakers_dir_out / f'{speaker}.txt'
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(utterances))
            self.log_and_manifest(
                stage='per_speaker_transcript',
                call_id=None,
                input_files=None,
                output_files=[str(transcript_path)],
                params={'speaker': speaker},
                metadata=None,
                event='file_written',
                result='success'
            )
        self.log_event('INFO', 'final_soundbites_complete', {'count': len(master_soundbites)})

    def get_master_transcript_path(self, call_id):
        """
        Returns the path to the canonical master transcript for a given call, suitable for LLM input.
        """
        soundbites_dir = self.run_folder / 'soundbites'
        return soundbites_dir / call_id / f"{call_id}_master_transcript.txt"

    def run_llm_task_for_call(self, call_id, master_transcript, llm_config, output_dir, llm_tasks, global_llm_seed=None):
        import requests
        import json
        output_paths = {}
        with open(master_transcript, 'r', encoding='utf-8') as f:
            transcript = f.read()
        base_url = llm_config.get('lm_studio_base_url', 'http://localhost:1234/v1')
        api_key = llm_config.get('lm_studio_api_key', 'lm-studio')
        model_id = llm_config.get('lm_studio_model_identifier', 'llama-3.1-8b-supernova-etherealhermes')
        temperature = llm_config.get('lm_studio_temperature', 0.5)
        max_tokens = llm_config.get('lm_studio_max_tokens', 1024)
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        for task in llm_tasks:
            name = task.get('name', 'unnamed_task')
            prompt_template = task.get('prompt_template', '')
            output_file = task.get('output_file', f'{name}.txt')
            prompt = prompt_template.format(transcript=transcript)
            # Determine seed
            if global_llm_seed is not None:
                # Deterministic per-task seed from global seed, call_id, and task name
                seed_input = f"{global_llm_seed}_{call_id}_{name}"
                seed = int(hashlib.sha256(seed_input.encode()).hexdigest(), 16) % (2**32)
            else:
                seed = random.randint(0, 2**32-1)
            data = {
                "model": model_id,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "seed": seed  # Pass seed if LLM API supports it
            }
            out_path = output_dir / output_file
            try:
                response = requests.post(f"{base_url}/chat/completions", headers=headers, data=json.dumps(data), timeout=60)
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content'].strip()
                    with open(out_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    output_paths[name] = str(out_path)
                else:
                    error_msg = f"LLM API error {response.status_code}: {response.text}"
                    with open(out_path, 'w', encoding='utf-8') as f:
                        f.write(error_msg)
                    output_paths[name] = str(out_path)
                    self.log_event('ERROR', 'llm_api_error', {'call_id': call_id, 'task': name, 'status': response.status_code, 'text': response.text, 'seed': seed})
            except Exception as e:
                with open(out_path, 'w', encoding='utf-8') as f:
                    f.write(f"LLM request failed: {e}")
                output_paths[name] = str(out_path)
                self.log_event('ERROR', 'llm_request_failed', {'call_id': call_id, 'task': name, 'error': str(e), 'seed': seed})
            # Always log the seed used
            self.log_event('INFO', 'llm_task_seed', {'call_id': call_id, 'task': name, 'seed': seed, 'output_file': str(out_path)})
        return output_paths

    def _get_global_context_seed(self, transcript_path, num_lines=10):
        """
        Read the first num_lines of the master transcript to use as a global context seed.
        """
        if not transcript_path or not os.path.exists(transcript_path):
            return ""
        with open(transcript_path, 'r', encoding='utf-8') as f:
            lines = []
            for _ in range(num_lines):
                line = f.readline()
                if not line:
                    break
                lines.append(line.strip())
        return '\n'.join(lines)

    def run_llm_stage(self, llm_config_path=None):
        import json
        from pathlib import Path
        # --- Read pipeline mode from pipeline_state.json if available ---
        mode = 'call'  # default
        pipeline_state_path = self.run_folder / 'pipeline_state.json'
        if pipeline_state_path.exists():
            try:
                with open(pipeline_state_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                if 'mode' in state:
                    mode = state['mode']
            except Exception as e:
                self.log_event('WARNING', 'pipeline_state_read_failed', {'error': str(e)})
        if llm_config_path is None:
            llm_config_path = Path('workflows/llm_tasks.json')
        else:
            llm_config_path = Path(llm_config_path)
        if not llm_config_path.exists():
            self.log_event('ERROR', 'llm_config_missing', {'path': str(llm_config_path)})
            return
        with open(llm_config_path, 'r', encoding='utf-8') as f:
            llm_config = json.load(f)
        llm_tasks = llm_config.get('llm_tasks', [])
        soundbites_dir = self.run_folder / 'soundbites'
        llm_dir = self.run_folder / 'llm'
        llm_dir.mkdir(exist_ok=True, parents=True)
        call_ids = [d.name for d in soundbites_dir.iterdir() if d.is_dir()]
        self.log_event('INFO', 'llm_stage_start', {'llm_dir': str(llm_dir), 'mode': mode})
        max_tokens = 16384
        safe_chunk = 10000  # chunk size for LLM input
        global_context_lines = 10  # Configurable: number of lines for global context seed
        chunk_overlap_lines = 3    # Configurable: number of lines to overlap between chunks
        for call_id in call_ids:
            master_transcript = self.get_master_transcript_path(call_id)
            transcript_text = None
            if master_transcript.exists():
                with open(master_transcript, 'r', encoding='utf-8') as f:
                    transcript_text = f.read()
            global_context_seed = self._get_global_context_seed(master_transcript, num_lines=global_context_lines)
            call_llm_dir = llm_dir / call_id
            call_llm_dir.mkdir(parents=True, exist_ok=True)
            if mode == 'call':
                # Run LLM tasks on the master transcript for each call
                for task in llm_tasks:
                    name = task.get('name', 'unnamed_task')
                    prompt_template = task.get('prompt_template', '')
                    output_file = f'{name}.txt'
                    output_path = call_llm_dir / output_file
                    # If transcript is too large, chunk it with overlap and prepend global context seed
                    if transcript_text and estimate_tokens(transcript_text) > safe_chunk:
                        lines = transcript_text.splitlines()
                        chunk_size = safe_chunk * 4 // 10  # Approximate lines per chunk
                        chunks = []
                        i = 0
                        while i < len(lines):
                            chunk_lines = lines[i:i+chunk_size]
                            # Add overlap from previous chunk
                            if i > 0:
                                chunk_lines = lines[max(0, i-chunk_overlap_lines):i+chunk_size]
                            chunk = '\n'.join(chunk_lines)
                            # Prepend global context seed
                            chunk = global_context_seed + '\n' + chunk
                            chunks.append(chunk)
                            i += chunk_size
                        responses = []
                        for idx, chunk in enumerate(chunks):
                            prompt = f"[Part {idx+1} of {len(chunks)}]\n" + prompt_template.format(transcript=chunk)
                            # ... existing LLM API call logic ...
                            import requests, random, hashlib
                            base_url = llm_config.get('lm_studio_base_url', 'http://localhost:1234/v1')
                            api_key = llm_config.get('lm_studio_api_key', 'lm-studio')
                            model_id = llm_config.get('lm_studio_model_identifier', 'llama-3.1-8b-supernova-etherealhermes')
                            temperature = llm_config.get('lm_studio_temperature', 0.5)
                            max_tokens = llm_config.get('lm_studio_max_tokens', 250)
                            headers = {
                                'Authorization': f'Bearer {api_key}',
                                'Content-Type': 'application/json'
                            }
                            seed_input = f"{call_id}_{name}_{idx}"
                            seed = int(hashlib.sha256(seed_input.encode()).hexdigest(), 16) % (2**32)
                            data = {
                                "model": model_id,
                                "messages": [
                                    {"role": "user", "content": prompt}
                                ],
                                "temperature": temperature,
                                "max_tokens": max_tokens,
                                "seed": seed
                            }
                            try:
                                response = requests.post(f"{base_url}/chat/completions", headers=headers, data=json.dumps(data), timeout=60)
                                if response.status_code == 200:
                                    result = response.json()
                                    content = result['choices'][0]['message']['content'].strip()
                                    responses.append(content)
                                else:
                                    error_msg = f"LLM API error {response.status_code}: {response.text}"
                                    responses.append(error_msg)
                                    self.log_event('ERROR', 'llm_api_error', {'call_id': call_id, 'task': name, 'status': response.status_code, 'text': response.text, 'seed': seed})
                            except Exception as e:
                                responses.append(f"LLM request failed: {e}")
                                self.log_event('ERROR', 'llm_request_failed', {'call_id': call_id, 'task': name, 'error': str(e), 'seed': seed})
                        final_response = '\n\n'.join(responses)
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(final_response)
                        self.log_and_manifest(
                            stage='llm',
                            call_id=call_id,
                            input_files=[str(master_transcript)],
                            output_files=[str(output_path)],
                            params={'task': name, 'prompt_template': prompt_template, 'chunks': len(chunks)},
                            metadata={'llm_model': llm_config.get('lm_studio_model_identifier')},
                            event='llm_task',
                            result='success'
                        )
                    else:
                        prompt = prompt_template.format(transcript=transcript_text)
                        # ... existing LLM API call logic ...
                        import requests, random, hashlib
                        base_url = llm_config.get('lm_studio_base_url', 'http://localhost:1234/v1')
                        api_key = llm_config.get('lm_studio_api_key', 'lm-studio')
                        model_id = llm_config.get('lm_studio_model_identifier', 'llama-3.1-8b-supernova-etherealhermes')
                        temperature = llm_config.get('lm_studio_temperature', 0.5)
                        max_tokens = llm_config.get('lm_studio_max_tokens', 250)
                        headers = {
                            'Authorization': f'Bearer {api_key}',
                            'Content-Type': 'application/json'
                        }
                        seed_input = f"{call_id}_{name}"
                        seed = int(hashlib.sha256(seed_input.encode()).hexdigest(), 16) % (2**32)
                        data = {
                            "model": model_id,
                            "messages": [
                                {"role": "user", "content": prompt}
                            ],
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "seed": seed
                        }
                        try:
                            response = requests.post(f"{base_url}/chat/completions", headers=headers, data=json.dumps(data), timeout=60)
                            if response.status_code == 200:
                                result = response.json()
                                content = result['choices'][0]['message']['content'].strip()
                                with open(output_path, 'w', encoding='utf-8') as f:
                                    f.write(content)
                            else:
                                error_msg = f"LLM API error {response.status_code}: {response.text}"
                                with open(output_path, 'w', encoding='utf-8') as f:
                                    f.write(error_msg)
                                self.log_event('ERROR', 'llm_api_error', {'call_id': call_id, 'task': name, 'status': response.status_code, 'text': response.text, 'seed': seed})
                        except Exception as e:
                            with open(output_path, 'w', encoding='utf-8') as f:
                                f.write(f"LLM request failed: {e}")
                            self.log_event('ERROR', 'llm_request_failed', {'call_id': call_id, 'task': name, 'error': str(e), 'seed': seed})
                        self.log_and_manifest(
                            stage='llm',
                            call_id=call_id,
                            input_files=[str(master_transcript)],
                            output_files=[str(output_path)],
                            params={'task': name, 'prompt_template': prompt_template},
                            metadata={'llm_model': llm_config.get('lm_studio_model_identifier')},
                            event='llm_task',
                            result='success'
                        )
                # Enable per-speaker LLM tasks for calls
                per_speaker = self.get_per_speaker_transcripts(call_id)
                if per_speaker:
                    speaker_outputs = {}
                    for speaker_id, speaker_text in per_speaker.items():
                        speaker_outputs[speaker_id] = {}
                        token_count = estimate_tokens(speaker_text)
                        if token_count > safe_chunk:
                            # Chunking logic (as above, with context seed and overlap)
                            lines = speaker_text.splitlines()
                            chunk_size = safe_chunk * 4 // 10
                            chunks = []
                            i = 0
                            while i < len(lines):
                                chunk_lines = lines[i:i+chunk_size]
                                if i > 0:
                                    chunk_lines = lines[max(0, i-chunk_overlap_lines):i+chunk_size]
                                chunk = global_context_seed + '\n' + '\n'.join(chunk_lines)
                                chunks.append(chunk)
                                i += chunk_size
                            responses = []
                            for idx, chunk in enumerate(chunks):
                                prompt = f"[Part {idx+1} of {len(chunks)} for Speaker {speaker_id}]\n" + prompt_template.format(transcript=chunk)
                                import requests, random, hashlib
                                base_url = llm_config.get('lm_studio_base_url', 'http://localhost:1234/v1')
                                api_key = llm_config.get('lm_studio_api_key', 'lm-studio')
                                model_id = llm_config.get('lm_studio_model_identifier', 'llama-3.1-8b-supernova-etherealhermes')
                                temperature = llm_config.get('lm_studio_temperature', 0.5)
                                max_tokens = llm_config.get('lm_studio_max_tokens', 250)
                                headers = {
                                    'Authorization': f'Bearer {api_key}',
                                    'Content-Type': 'application/json'
                                }
                                seed_input = f"{call_id}_{speaker_id}_{name}_{idx}"
                                seed = int(hashlib.sha256(seed_input.encode()).hexdigest(), 16) % (2**32)
                                data = {
                                    "model": model_id,
                                    "messages": [
                                        {"role": "user", "content": prompt}
                                    ],
                                    "temperature": temperature,
                                    "max_tokens": max_tokens,
                                    "seed": seed
                                }
                                try:
                                    response = requests.post(f"{base_url}/chat/completions", headers=headers, data=json.dumps(data), timeout=60)
                                    if response.status_code == 200:
                                        result = response.json()
                                        content = result['choices'][0]['message']['content'].strip()
                                        responses.append(content)
                                    else:
                                        error_msg = f"LLM API error {response.status_code}: {response.text}"
                                        responses.append(error_msg)
                                        self.log_event('ERROR', 'llm_api_error', {'call_id': call_id, 'speaker_id': speaker_id, 'task': name, 'status': response.status_code, 'text': response.text, 'seed': seed})
                                except Exception as e:
                                    responses.append(f"LLM request failed: {e}")
                                    self.log_event('ERROR', 'llm_request_failed', {'call_id': call_id, 'speaker_id': speaker_id, 'task': name, 'error': str(e), 'seed': seed})
                            final_response = '\n\n'.join(responses)
                            output_file = f'{speaker_id}_{name}.txt'
                            output_path = call_llm_dir / output_file
                            with open(output_path, 'w', encoding='utf-8') as f:
                                f.write(final_response)
                            speaker_outputs[speaker_id][name] = str(output_path)
                            self.log_and_manifest(
                                stage='llm',
                                call_id=call_id,
                                input_files=[f'per-speaker:{speaker_id}'],
                                output_files=[str(output_path)],
                                params={'task': name, 'prompt_template': prompt_template, 'speaker_id': speaker_id, 'chunks': len(chunks)},
                                metadata={'llm_model': llm_config.get('lm_studio_model_identifier')},
                                event='llm_task',
                                result='success'
                            )
                        else:
                            prompt = prompt_template.format(transcript=speaker_text)
                            output_file = f'{speaker_id}_{name}.txt'
                            output_path = call_llm_dir / output_file
                            import requests, random, hashlib
                            base_url = llm_config.get('lm_studio_base_url', 'http://localhost:1234/v1')
                            api_key = llm_config.get('lm_studio_api_key', 'lm-studio')
                            model_id = llm_config.get('lm_studio_model_identifier', 'llama-3.1-8b-supernova-etherealhermes')
                            temperature = llm_config.get('lm_studio_temperature', 0.5)
                            max_tokens = llm_config.get('lm_studio_max_tokens', 250)
                            headers = {
                                'Authorization': f'Bearer {api_key}',
                                'Content-Type': 'application/json'
                            }
                            seed_input = f"{call_id}_{speaker_id}_{name}"
                            seed = int(hashlib.sha256(seed_input.encode()).hexdigest(), 16) % (2**32)
                            data = {
                                "model": model_id,
                                "messages": [
                                    {"role": "user", "content": prompt}
                                ],
                                "temperature": temperature,
                                "max_tokens": max_tokens,
                                "seed": seed
                            }
                            try:
                                response = requests.post(f"{base_url}/chat/completions", headers=headers, data=json.dumps(data), timeout=60)
                                if response.status_code == 200:
                                    result = response.json()
                                    content = result['choices'][0]['message']['content'].strip()
                                    with open(output_path, 'w', encoding='utf-8') as f:
                                        f.write(content)
                                else:
                                    error_msg = f"LLM API error {response.status_code}: {response.text}"
                                    with open(output_path, 'w', encoding='utf-8') as f:
                                        f.write(error_msg)
                                    self.log_event('ERROR', 'llm_api_error', {'call_id': call_id, 'speaker_id': speaker_id, 'task': name, 'status': response.status_code, 'text': response.text, 'seed': seed})
                            except Exception as e:
                                with open(output_path, 'w', encoding='utf-8') as f:
                                    f.write(f"LLM request failed: {e}")
                                self.log_event('ERROR', 'llm_request_failed', {'call_id': call_id, 'speaker_id': speaker_id, 'task': name, 'error': str(e), 'seed': seed})
                            speaker_outputs[speaker_id][name] = str(output_path)
                            self.log_and_manifest(
                                stage='llm',
                                call_id=call_id,
                                input_files=[f'per-speaker:{speaker_id}'],
                                output_files=[str(output_path)],
                                params={'task': name, 'prompt_template': prompt_template, 'speaker_id': speaker_id},
                                metadata={'llm_model': llm_config.get('lm_studio_model_identifier')},
                                event='llm_task',
                                result='success'
                            )
                    # Aggregate per-speaker outputs into a summary file
                    agg_path = call_llm_dir / 'per_speaker_llm_outputs.json'
                    with open(agg_path, 'w', encoding='utf-8') as f:
                        json.dump(speaker_outputs, f, indent=2)
                    self.log_and_manifest(
                        stage='llm',
                        call_id=call_id,
                        input_files=None,
                        output_files=[str(agg_path)],
                        params={'aggregation': 'per-speaker'},
                        metadata={'speaker_outputs': speaker_outputs},
                        event='llm_aggregation',
                        result='success'
                    )
            else:
                # Single-file mode: run per-speaker LLM tasks as before
                per_speaker = self.get_per_speaker_transcripts(call_id)
                speaker_outputs = {}
                for speaker_id, speaker_text in per_speaker.items():
                    speaker_outputs[speaker_id] = {}
                    token_count = estimate_tokens(speaker_text)
                    if token_count > safe_chunk:
                        # Chunking logic (as before)
                        words = speaker_text.split()
                        chunk_size = safe_chunk * 4
                        chunks = []
                        current = []
                        current_len = 0
                        for word in words:
                            current.append(word)
                            current_len += len(word) + 1
                            if current_len >= chunk_size:
                                chunks.append(' '.join(current))
                                current = []
                                current_len = 0
                        if current:
                            chunks.append(' '.join(current))
                        for task in llm_tasks:
                            name = task.get('name', 'unnamed_task')
                            prompt_template = task.get('prompt_template', '')
                            output_file = f'{speaker_id}_{name}.txt'
                            output_path = call_llm_dir / output_file
                            responses = []
                            for i, chunk in enumerate(chunks):
                                prompt = f"[Part {i+1} of {len(chunks)} for Speaker {speaker_id}]\n" + prompt_template.format(transcript=chunk)
                                import requests, random, hashlib
                                base_url = llm_config.get('lm_studio_base_url', 'http://localhost:1234/v1')
                                api_key = llm_config.get('lm_studio_api_key', 'lm-studio')
                                model_id = llm_config.get('lm_studio_model_identifier', 'llama-3.1-8b-supernova-etherealhermes')
                                temperature = llm_config.get('lm_studio_temperature', 0.5)
                                max_tokens = llm_config.get('lm_studio_max_tokens', 250)
                                headers = {
                                    'Authorization': f'Bearer {api_key}',
                                    'Content-Type': 'application/json'
                                }
                                seed_input = f"{call_id}_{speaker_id}_{name}_{i}"
                                seed = int(hashlib.sha256(seed_input.encode()).hexdigest(), 16) % (2**32)
                                data = {
                                    "model": model_id,
                                    "messages": [
                                        {"role": "user", "content": prompt}
                                    ],
                                    "temperature": temperature,
                                    "max_tokens": max_tokens,
                                    "seed": seed
                                }
                                try:
                                    response = requests.post(f"{base_url}/chat/completions", headers=headers, data=json.dumps(data), timeout=60)
                                    if response.status_code == 200:
                                        result = response.json()
                                        content = result['choices'][0]['message']['content'].strip()
                                        responses.append(content)
                                    else:
                                        error_msg = f"LLM API error {response.status_code}: {response.text}"
                                        responses.append(error_msg)
                                        self.log_event('ERROR', 'llm_api_error', {'call_id': call_id, 'speaker_id': speaker_id, 'task': name, 'status': response.status_code, 'text': response.text, 'seed': seed})
                                except Exception as e:
                                    responses.append(f"LLM request failed: {e}")
                                    self.log_event('ERROR', 'llm_request_failed', {'call_id': call_id, 'speaker_id': speaker_id, 'task': name, 'error': str(e), 'seed': seed})
                            final_response = '\n\n'.join(responses)
                            with open(output_path, 'w', encoding='utf-8') as f:
                                f.write(final_response)
                            speaker_outputs[speaker_id][name] = str(output_path)
                            self.log_and_manifest(
                                stage='llm',
                                call_id=call_id,
                                input_files=[f'per-speaker:{speaker_id}'],
                                output_files=[str(output_path)],
                                params={'task': name, 'prompt_template': prompt_template, 'speaker_id': speaker_id, 'chunks': len(chunks)},
                                metadata={'llm_model': llm_config.get('lm_studio_model_identifier')},
                                event='llm_task',
                                result='success'
                            )
                    else:
                        for task in llm_tasks:
                            name = task.get('name', 'unnamed_task')
                            prompt_template = task.get('prompt_template', '')
                            output_file = f'{speaker_id}_{name}.txt'
                            output_path = call_llm_dir / output_file
                            prompt = prompt_template.format(transcript=speaker_text)
                            import requests, random, hashlib
                            base_url = llm_config.get('lm_studio_base_url', 'http://localhost:1234/v1')
                            api_key = llm_config.get('lm_studio_api_key', 'lm-studio')
                            model_id = llm_config.get('lm_studio_model_identifier', 'llama-3.1-8b-supernova-etherealhermes')
                            temperature = llm_config.get('lm_studio_temperature', 0.5)
                            max_tokens = llm_config.get('lm_studio_max_tokens', 250)
                            headers = {
                                'Authorization': f'Bearer {api_key}',
                                'Content-Type': 'application/json'
                            }
                            seed_input = f"{call_id}_{speaker_id}_{name}"
                            seed = int(hashlib.sha256(seed_input.encode()).hexdigest(), 16) % (2**32)
                            data = {
                                "model": model_id,
                                "messages": [
                                    {"role": "user", "content": prompt}
                                ],
                                "temperature": temperature,
                                "max_tokens": max_tokens,
                                "seed": seed
                            }
                            try:
                                response = requests.post(f"{base_url}/chat/completions", headers=headers, data=json.dumps(data), timeout=60)
                                if response.status_code == 200:
                                    result = response.json()
                                    content = result['choices'][0]['message']['content'].strip()
                                    with open(output_path, 'w', encoding='utf-8') as f:
                                        f.write(content)
                                else:
                                    error_msg = f"LLM API error {response.status_code}: {response.text}"
                                    with open(output_path, 'w', encoding='utf-8') as f:
                                        f.write(error_msg)
                                    self.log_event('ERROR', 'llm_api_error', {'call_id': call_id, 'speaker_id': speaker_id, 'task': name, 'status': response.status_code, 'text': response.text, 'seed': seed})
                            except Exception as e:
                                with open(output_path, 'w', encoding='utf-8') as f:
                                    f.write(f"LLM request failed: {e}")
                                self.log_event('ERROR', 'llm_request_failed', {'call_id': call_id, 'speaker_id': speaker_id, 'task': name, 'error': str(e), 'seed': seed})
                            speaker_outputs[speaker_id][name] = str(output_path)
                            self.log_and_manifest(
                                stage='llm',
                                call_id=call_id,
                                input_files=[f'per-speaker:{speaker_id}'],
                                output_files=[str(output_path)],
                                params={'task': name, 'prompt_template': prompt_template, 'speaker_id': speaker_id},
                                metadata={'llm_model': llm_config.get('lm_studio_model_identifier')},
                                event='llm_task',
                                result='success'
                            )
                # Aggregate per-speaker outputs into a summary file
                agg_path = call_llm_dir / 'per_speaker_llm_outputs.json'
                with open(agg_path, 'w', encoding='utf-8') as f:
                    json.dump(speaker_outputs, f, indent=2)
                self.log_and_manifest(
                    stage='llm',
                    call_id=call_id,
                    input_files=None,
                    output_files=[str(agg_path)],
                    params={'aggregation': 'per-speaker'},
                    metadata={'speaker_outputs': speaker_outputs},
                    event='llm_aggregation',
                    result='success'
                )

    def run_normalization_stage(self):
        """
        Normalize separated vocal stems to -14.0 LUFS and output to normalized/<call id>/<channel>.wav.
        Downstream stages use normalized vocals.
        Logs every file written and updates manifest.
        """
        import pyloudnorm as pyln
        import soundfile as sf
        from pathlib import Path
        separated_dir = self.run_folder / 'separated'
        normalized_dir = self.run_folder / 'normalized'
        normalized_dir.mkdir(exist_ok=True)
        meter = pyln.Meter(44100)
        self.log_event('INFO', 'normalization_start', {'dir': str(separated_dir)})
        for call_id in os.listdir(separated_dir):
            call_sep_dir = separated_dir / call_id
            if not call_sep_dir.is_dir():
                continue
            call_norm_dir = normalized_dir / call_id
            call_norm_dir.mkdir(parents=True, exist_ok=True)
            for channel in ['left-vocals', 'right-vocals']:
                src = call_sep_dir / f"{channel}.wav"
                if not src.exists():
                    continue
                audio, sr = sf.read(str(src))
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != 44100:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=44100)
                    sr = 44100
                loudness = meter.integrated_loudness(audio)
                audio_norm = pyln.normalize.loudness(audio, loudness, -14.0)
                out_path = call_norm_dir / f"{channel}.wav"
                sf.write(str(out_path), audio_norm, sr)
                self.log_and_manifest(
                    stage='normalized',
                    call_id=call_id,
                    input_files=[str(src)],
                    output_files=[str(out_path)],
                    params={'target_lufs': -14.0},
                    metadata={'measured_lufs': loudness},
                    event='file_written',
                    result='success'
                )
        self.log_event('INFO', 'normalization_complete', {'normalized_dir': str(normalized_dir)})

    def run_true_peak_normalization_stage(self):
        """
        Apply true peak normalization to prevent digital clipping on normalized vocals.
        Uses -1.0 dBTP (decibels True Peak) limit which is broadcast standard.
        Logs every file written and updates manifest.
        """
        import pyloudnorm as pyln
        import soundfile as sf
        from pathlib import Path
        
        normalized_dir = self.run_folder / 'normalized'
        true_peak_dir = self.run_folder / 'true_peak_normalized'
        true_peak_dir.mkdir(exist_ok=True)
        
        self.log_event('INFO', 'true_peak_normalization_start', {'target_dbtp': -1.0})
        
        for call_id in os.listdir(normalized_dir):
            call_norm_dir = normalized_dir / call_id
            if not call_norm_dir.is_dir():
                continue
            
            call_tp_dir = true_peak_dir / call_id
            call_tp_dir.mkdir(parents=True, exist_ok=True)
            
            # Process all vocal files (both traditional and conversation)
            for vocal_file in call_norm_dir.glob('*.wav'):
                src = vocal_file
                dst = call_tp_dir / vocal_file.name
                
                try:
                    audio, sr = sf.read(str(src))
                    
                    # Ensure mono for processing
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    
                    # Apply true peak limiting to -1.0 dBTP
                    meter = pyln.Meter(sr)
                    peak_normalized = pyln.normalize.peak(audio, -1.0)
                    
                    # Save the true peak normalized audio
                    sf.write(str(dst), peak_normalized, sr)
                    
                    # Calculate true peak level for logging
                    true_peak_db = 20 * np.log10(np.max(np.abs(peak_normalized)))
                    
                    self.log_and_manifest(
                        stage='true_peak_normalized',
                        call_id=call_id,
                        input_files=[str(src)],
                        output_files=[str(dst)],
                        params={'target_dbtp': -1.0},
                        metadata={'measured_true_peak_db': true_peak_db},
                        event='file_written',
                        result='success'
                    )
                    
                except Exception as e:
                    self.log_event('ERROR', 'true_peak_normalization_failed', {
                        'call_id': call_id,
                        'file': vocal_file.name,
                        'error': str(e)
                    })
        
        self.log_event('INFO', 'true_peak_normalization_complete', {'true_peak_dir': str(true_peak_dir)})

    def run_remix_stage(self, call_tones=False):
        """
        For each segment, mix true peak normalized vocals (or normalized if not available) into stereo (duplicate channel if mono).
        Output stereo remixed_segment.wav for each segment/job index.
        Logs every file written and updates manifest.
        """
        import soundfile as sf
        import numpy as np
        from pathlib import Path
        true_peak_dir = self.run_folder / 'true_peak_normalized'
        normalized_dir = self.run_folder / 'normalized'
        vocals_dir = true_peak_dir if true_peak_dir.exists() else normalized_dir
        remix_dir = self.run_folder / 'remix'
        remix_dir.mkdir(exist_ok=True)
        self.log_event('INFO', 'remix_start', {
            'vocals_source': 'true_peak_normalized' if vocals_dir == true_peak_dir else 'normalized',
            'vocals_dir': str(vocals_dir)
        })
        for seg_dir in vocals_dir.iterdir():
            if not seg_dir.is_dir():
                continue
            for wav_file in seg_dir.glob('*.wav'):
                audio, sr = sf.read(str(wav_file))
                if audio.ndim == 1:
                    stereo = np.stack([audio, audio], axis=-1)
                elif audio.ndim == 2 and audio.shape[1] == 1:
                    stereo = np.repeat(audio, 2, axis=1)
                else:
                    stereo = audio
                idx = seg_dir.name
                out_dir = remix_dir / idx
                out_dir.mkdir(parents=True, exist_ok=True)
                remixed_path = out_dir / 'remixed_segment.wav'
                sf.write(str(remixed_path), stereo, sr)
                self.log_and_manifest(
                    stage='remix',
                    call_id=None,
                    input_files=[str(wav_file)],
                    output_files=[str(remixed_path)],
                    params={'panning': 'mono to stereo'},
                    metadata={'vocals_source': 'true_peak_normalized' if vocals_dir == true_peak_dir else 'normalized'},
                    event='file_written',
                    result='success'
                )
        self.log_event('INFO', 'remix_complete', {'remix_dir': str(remix_dir)})

    def run_show_stage(self, call_tones=False):
        """
        Concatenate all remixed segments into show/show.wav (44.1kHz, stereo, 16-bit).
        Write show/show.json with start/end times for each segment, plus metadata.
        Logs every file written and updates manifest.
        """
        import soundfile as sf
        import numpy as np
        from pathlib import Path
        show_dir = self.run_folder / 'show'
        show_dir.mkdir(exist_ok=True)
        remix_dir = self.run_folder / 'remix'
        show_wav_path = show_dir / 'show.wav'
        show_json_path = show_dir / 'show.json'
        segment_files = []
        self.log_event('INFO', 'show_start', {'remix_dir': str(remix_dir)})
        for idx in sorted(os.listdir(remix_dir)):
            remixed_path = remix_dir / idx / 'remixed_segment.wav'
            if remixed_path.exists():
                segment_files.append((idx, remixed_path))
        show_audio = []
        show_timeline = []
        sr = 44100
        cur_time = 0.0
        for idx, seg_path in segment_files:
            audio, file_sr = sf.read(str(seg_path))
            if file_sr != sr:
                import librosa
                audio = librosa.resample(audio.T, orig_sr=file_sr, target_sr=sr).T
            start = cur_time
            end = start + audio.shape[0] / sr
            show_audio.append(audio)
            show_timeline.append({
                'segment_index': idx,
                'start': start,
                'end': end
            })
            cur_time = end
        if show_audio:
            show_audio = np.concatenate(show_audio, axis=0)
            sf.write(str(show_wav_path), show_audio, sr, subtype='PCM_16')
            self.log_and_manifest(
                stage='show',
                call_id=None,
                input_files=[str(f[1]) for f in segment_files],
                output_files=[str(show_wav_path)],
                params={},
                metadata={'timeline': show_timeline},
                event='file_written',
                result='success'
            )
            with open(show_json_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(show_timeline, f, indent=2)
            self.log_and_manifest(
                stage='show',
                call_id=None,
                input_files=None,
                output_files=[str(show_json_path)],
                params=None,
                metadata=None,
                event='file_written',
                result='success'
            )
        self.log_event('INFO', 'show_complete', {'show_wav': str(show_wav_path), 'show_json': str(show_json_path)})

    def run_finalization_stage(self):
        """
        Run the finalization stage - converts to MP3, embeds ID3 tags, handles LLM output
        """
        run_finalization_stage(self.run_folder, self.manifest)

    def run(self, mode='auto', call_cutter=False, call_tones=False):
        print(f"[INFO] Running pipeline in {mode} mode.")
        if mode == 'single-file':
            self.run_single_file_workflow(call_tones=call_tones)
        elif mode == 'calls':
            self.run_call_processing_workflow(call_tones=call_tones)
        else:
            self.run_call_processing_workflow(call_tones=call_tones)

    def run_single_file_workflow(self, call_tones=False):
        # Full pipeline for single-file mode
        self._run_ingestion_jobs()
        # Find the original input audio file (should be in renamed/)
        renamed_dir = self.run_folder / 'renamed'
        input_files = [f for f in renamed_dir.iterdir() if f.is_file()]
        if not input_files:
            print("[ERROR] No input files found in renamed/ directory.")
            return
        original_audio = input_files[0]
        print(f"[DEBUG] Using original audio for CLAP: {original_audio}")
        # Run CLAP segmentation on original audio
        segmented_dir = self.run_folder / 'segmented'
        segmented_dir.mkdir(parents=True, exist_ok=True)
        from clap_annotation import segment_audio_with_clap
        import json
        segmentation_config_path = Path('workflows/clap_segmentation.json')
        with open(segmentation_config_path, 'r', encoding='utf-8') as f:
            segmentation_config = json.load(f)["clap_segmentation"]
        segments = segment_audio_with_clap(
            original_audio,
            segmentation_config,
            segmented_dir,
            model_id=segmentation_config.get("model_id", "laion/clap-htsat-unfused"),
            chunk_length_sec=segmentation_config.get("chunk_length_sec", 5),
            overlap_sec=segmentation_config.get("overlap_sec", 2)
        )
        # If segments found, process each segment; else, process the whole file
        files_to_process = []
        if segments:
            print(f"[DEBUG] CLAP segmentation found {len(segments)} segments.")
            for seg in segments:
                files_to_process.append(Path(seg["output_path"]))
        else:
            print("[DEBUG] No CLAP segments found; processing whole file.")
            files_to_process = [original_audio]
        # For each file (segment or whole), perform separation and downstream steps
        for idx, audio_file in enumerate(files_to_process):
            print(f"[DEBUG] Processing segment {idx}: {audio_file}")
            # Place segment in a per-segment input folder for separation
            seg_input_dir = self.run_folder / 'segment_inputs' / f'segment_{idx:04d}'
            seg_input_dir.mkdir(parents=True, exist_ok=True)
            seg_audio_path = seg_input_dir / audio_file.name
            import shutil
            shutil.copy2(audio_file, seg_audio_path)
            # Ingest segment as a job
            job = Job(job_id=f'segment_{idx:04d}', data={
                'orig_path': str(seg_audio_path),
                'base_name': seg_audio_path.name,
                'is_tuple': False,
                'tuple_index': idx,
                'subid': 'c',
                'file_type': 'out',
                'timestamp': '',
                'ext': seg_audio_path.suffix
            })
            self.jobs = [job]  # Overwrite jobs for this segment
            self.add_separation_jobs()
            self.run_audio_separation_stage()
            self.run_normalization_stage()
            self.run_true_peak_normalization_stage()
            # Use separated vocals for downstream steps
            separated_dir = self.run_folder / 'separated' / f'{idx:04d}'
            vocal_path = separated_dir / 'out-vocals.wav'
            if not vocal_path.exists():
                wavs = list(separated_dir.glob('*.wav'))
                if not wavs:
                    print(f"[ERROR] No separated vocals found for segment {idx}.")
                    continue
                vocal_path = wavs[0]
            # Run CLAP annotation on the separated vocal (for context, not segmentation)
            from clap_annotation import annotate_clap_for_out_files
            clap_dir = self.run_folder / 'clap' / f'{idx:04d}'
            annotate_clap_for_out_files(
                input_dir=Path(vocal_path).parent,
                output_dir=clap_dir,
                prompts=segmentation_config.get("prompts", []),
                model_id=segmentation_config.get("model_id", "laion/clap-htsat-unfused"),
                chunk_length_sec=segmentation_config.get("chunk_length_sec", 5),
                overlap_sec=segmentation_config.get("overlap_sec", 2),
                confidence_threshold=segmentation_config.get("confidence_threshold", 0.6)
            )
            # Downstream steps
            self.run_diarization_stage()
            self.run_speaker_segmentation_stage()
            self.run_resample_segments_stage()
            self.run_transcription_stage(asr_engine=self.asr_engine)
            self.run_rename_soundbites_stage()
            self.run_final_soundbite_stage()
            self.run_llm_stage(llm_config_path=self.llm_config_path)
            self.run_remix_stage(call_tones=call_tones)
            self.run_show_stage(call_tones=call_tones)
            self.run_finalization_stage()

    def run_call_processing_workflow(self, call_tones=False):
        # Full pipeline for call-processing mode
        self._run_ingestion_jobs()
        self.add_separation_jobs()
        self.run_audio_separation_stage()
        self.run_normalization_stage()
        self.run_true_peak_normalization_stage()
        self.run_diarization_stage()
        self.run_speaker_segmentation_stage()
        self.run_resample_segments_stage()
        self.run_transcription_stage(asr_engine=self.asr_engine)
        self.run_rename_soundbites_stage()
        self.run_final_soundbite_stage()
        self.run_llm_stage(llm_config_path=self.llm_config_path)
        self.run_remix_stage(call_tones=call_tones)
        self.run_show_stage(call_tones=call_tones)
        self.run_finalization_stage()

    def run_with_resume(self, call_tones=False, resume=True, resume_from=None):
        # Add resume functionality to orchestrator
        add_resume_to_orchestrator(self, resume_mode=True)
        # Print resume summary at start
        print_resume_status(self.run_folder, detailed=True)
        # Define all stages with their methods
        stages_and_methods = [
            ('ingestion', self._run_ingestion_jobs),
            ('separation', lambda: (self.add_separation_jobs(), self.run_audio_separation_stage())),
            ('normalization', self.run_normalization_stage),
            ('true_peak_normalization', self.run_true_peak_normalization_stage),
            ('diarization', self.run_diarization_stage),
            ('segmentation', self.run_speaker_segmentation_stage),
            ('resampling', self.run_resample_segments_stage),
            ('transcription', lambda: self.run_transcription_stage(asr_engine=self.asr_engine)),
            ('soundbite_renaming', self.run_rename_soundbites_stage),
            ('soundbite_finalization', self.run_final_soundbite_stage),
            ('llm', self.run_llm_stage),
            ('remix', lambda: self.run_remix_stage(call_tones=call_tones)),
            ('show', lambda: self.run_show_stage(call_tones=call_tones)),
            ('finalization', self.run_finalization_stage)
        ]
        # Run each stage with resume support
        for stage_name, stage_method in stages_and_methods:
            try:
                run_stage_with_resume(self, stage_name, stage_method, resume_from)
            except Exception as e:
                self.log_event('ERROR', f'pipeline_failed_at_stage', {
                    'stage': stage_name,
                    'error': str(e)
                })
                console.print(f"\n[bold red]Pipeline failed at stage: {stage_name}[/]")
                console.print(f"[red]Error: {e}[/]")
                console.print(f"\n[yellow]To resume from this point, run with --resume[/]")
                raise
        # Write final manifest and log
        self.write_manifest()
        self.write_log()
        # Final summary
        print("\n\033[92m🎉🎉 Pipeline completed successfully! All outputs are ready. 🎉🎉\033[0m\n")
        print_resume_status(self.run_folder, detailed=True)

    def _run_ingestion_jobs(self):
        """Helper method for ingestion stage that can be used with resume functionality"""
        
        # Debug: Log job information
        total_jobs = len(self.jobs)
        rename_jobs = [job for job in self.jobs if job.job_type == 'rename']
        
        self.log_event('INFO', 'ingestion_start', {
            'total_jobs': total_jobs,
            'rename_jobs': len(rename_jobs),
            'job_details': [{'job_id': j.job_id, 'job_type': j.job_type} for j in self.jobs[:5]]  # First 5 jobs, anonymized
        })
        
        if total_jobs == 0:
            self.log_event('WARNING', 'no_jobs_to_process', {'message': 'No ingestion jobs found'})
            return
        
        with tqdm(total=len(self.jobs), desc="Ingestion Progress", position=0) as global_bar:
            processed_files = 0
            for job in self.jobs:
                if job.job_type == 'rename':
                    result = process_file_job(job, self.run_folder)
                    
                    # Debug: Log each file processing result
                    self.log_event('INFO', 'file_processing_result', {
                        'job_id': job.job_id,
                        'success': result.get('success', False),
                        'output_name': result.get('output_name'),
                        'output_path': result.get('output_path'),
                        'error': result.get('error')
                    })
                    
                    manifest_entry = {
                        'job_id': job.job_id,
                        'tuple_index': job.data.get('tuple_index'),
                        'subid': job.data.get('subid'),
                        'type': job.data.get('file_type'),
                        'timestamp': job.data.get('timestamp'),
                        'output_name': result.get('output_name'),
                        'output_path': result.get('output_path'),
                        'stage': 'renamed',
                    }
                    if result['success']:
                        self.log_event('INFO', 'file_renamed', {
                            'output_name': result.get('output_name'),
                            'output_path': result.get('output_path')
                        })
                        self.manifest.append(manifest_entry)
                        processed_files += 1
                    else:
                        self.log_event('ERROR', 'file_rename_failed', {
                            'output_name': result.get('output_name'),
                            'output_path': result.get('output_path'),
                            'error': result.get('error')
                        })
                        self.manifest.append(manifest_entry)
                    global_bar.update(1)
        
        self.log_event('INFO', 'ingestion_complete', {
            'total_jobs': total_jobs,
            'processed_files': processed_files,
            'renamed_dir': str(self.run_folder / 'renamed')
        })
        
        self.enable_logging()
        self.write_log()
        self.write_manifest()

    def log_and_manifest(self, stage, call_id=None, input_files=None, output_files=None, params=None, metadata=None, event='file_written', result='success', error=None):
        """
        Helper to log and add manifest entry for any file operation.
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'stage': stage,
            'call_id': call_id,
            'input_files': input_files,
            'output_files': output_files,
            'params': params,
            'metadata': metadata,
            'result': result,
            'error': error
        }
        self.log_event('INFO' if result == 'success' else 'ERROR', event, entry)
        manifest_entry = {
            'stage': stage,
            'call_id': call_id,
            'input_files': input_files,
            'output_files': output_files,
            'params': params,
            'metadata': metadata,
            'result': result,
            'error': error
        }
        self.manifest.append(manifest_entry)

    def run_clap_segmentation_stage(self, segmentation_config_path=None):
        """
        Run CLAP-based segmentation for long audio files if --call-cutter is set.
        Loads config from workflows/clap_segmentation.json by default.
        """
        renamed_dir = self.run_folder / 'renamed'
        segmented_dir = self.run_folder / 'segmented'
        segmented_dir.mkdir(exist_ok=True)
        # Load config
        import json
        config_path = segmentation_config_path or Path('workflows/clap_segmentation.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            segmentation_config = json.load(f)["clap_segmentation"]
        # Find eligible files (e.g., out- files longer than min_segment_length)
        min_len = segmentation_config.get("min_segment_length_sec", 10)
        for file in renamed_dir.iterdir():
            if not file.is_file() or '-out-' not in file.name:
                continue
            import soundfile as sf
            try:
                info = sf.info(str(file))
                duration = info.duration
            except Exception:
                continue
            if duration < min_len:
                continue
            call_id, channel, timestamp = file.stem.split('-')[0], 'out', None
            out_dir = segmented_dir / call_id
            out_dir.mkdir(parents=True, exist_ok=True)
            segments = segment_audio_with_clap(file, segmentation_config, out_dir)
            for seg in segments:
                self.log_and_manifest(
                    stage='clap_segmented',
                    call_id=call_id,
                    input_files=[str(file)],
                    output_files=[seg["output_path"]],
                    params={'start': seg['start'], 'end': seg['end']},
                    metadata={'events': seg['events']},
                    event='clap_segment',
                    result='success'
                )
                self.manifest.append({
                    'stage': 'clap_segmented',
                    'call_id': call_id,
                    'input_name': file.name,
                    'segment_index': seg['segment_index'],
                    'start': seg['start'],
                    'end': seg['end'],
                    'output_path': seg['output_path'],
                    'events': seg['events']
                })
        self.log_event('INFO', 'clap_segmentation_complete', {'segmented_dir': str(segmented_dir)})

    def get_per_speaker_transcripts(self, call_id):
        """
        Extract per-speaker transcripts for a call from speakers/ and diarized/ folders.
        Returns a dict: {speaker_id: transcript_text}
        """
        speakers_dir = self.run_folder / 'speakers' / call_id
        transcripts = {}
        if not speakers_dir.exists():
            return transcripts
        for channel_dir in speakers_dir.iterdir():
            if not channel_dir.is_dir():
                continue
            for speaker_dir in channel_dir.iterdir():
                if not speaker_dir.is_dir():
                    continue
                speaker_id = speaker_dir.name
                utterances = []
                for txt_file in sorted(speaker_dir.glob('*.txt')):
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        if text:
                            utterances.append(text)
                if utterances:
                    transcripts[speaker_id] = '\n'.join(utterances)
        return transcripts

def create_jobs_from_input(input_path: Path) -> List[Job]:
    """Create jobs from input path (can be a single file or directory)
    IMPORTANT: No PII (original filenames or paths) may be printed or logged here! Only output anonymized counts if needed.
    """
    all_files = []
    # Handle both single files and directories
    if input_path.is_file():
        file = input_path.name
        ext = input_path.suffix.lower()
        if ext in SUPPORTED_EXTENSIONS:
            orig_path = input_path
            base_name = file
            all_files.append((orig_path, base_name))
        elif ext in VIDEO_EXTENSIONS:
            temp_audio_dir = Path('outputs') / f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}/temp_audio"
            temp_audio_dir.mkdir(parents=True, exist_ok=True)
            audio_name = Path(file).stem + '.wav'
            audio_path = temp_audio_dir / audio_name
            print(f"[INFO] Extracting audio from video file: {file}")
            try:
                subprocess.run([
                    'ffmpeg', '-y', '-i', str(input_path),
                    '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', str(audio_path)
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(f"[INFO] Audio extracted to: {audio_path}")
                all_files.append((audio_path, audio_name))
            except Exception as e:
                print(f"[ERROR] Failed to extract audio from {file}: {e}")
    elif input_path.is_dir():
        files_found = list(input_path.iterdir())
        print(f"[DEBUG] Files found in input_dir: {[str(f) for f in files_found]}")
        url_wav = input_path / 'input_from_url.wav'
        if url_wav.exists():
            print(f"[DEBUG] Found input_from_url.wav, adding as single-file job.")
            all_files.append((url_wav, url_wav.name))
        for file in files_found:
            ext = file.suffix.lower()
            orig_path = file
            base_name = file.name
            if ext in SUPPORTED_EXTENSIONS and file.name != 'input_from_url.wav':
                all_files.append((orig_path, base_name))
            elif ext in VIDEO_EXTENSIONS:
                run_ts = datetime.now().strftime('%Y%m%d-%H%M%S')
                temp_audio_dir = Path('outputs') / f"run-{run_ts}/temp_audio"
                temp_audio_dir.mkdir(parents=True, exist_ok=True)
                audio_name = Path(file).stem + '.wav'
                audio_path = temp_audio_dir / audio_name
                print(f"[INFO] Extracting audio from video file: {file}")
                try:
                    subprocess.run([
                        'ffmpeg', '-y', '-i', str(orig_path),
                        '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', str(audio_path)
                    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    print(f"[INFO] Audio extracted to: {audio_path}")
                    all_files.append((audio_path, audio_name))
                except Exception as e:
                    print(f"[ERROR] Failed to extract audio from {file}: {e}")
    else:
        print(f"❌ Input path does not exist or is invalid.")
        return []
    if not all_files:
        print(f"❌ No supported audio files found in input.")
        print(f"Supported extensions: {SUPPORTED_EXTENSIONS}")
        return []
    print(f"🔧 Created {len(all_files)} job(s) for processing.")
    jobs = []
    for idx, (orig_path, base_name) in enumerate(all_files):
        ext = Path(base_name).suffix.lower()
        ts = os.path.getmtime(orig_path)
        ts_str = datetime.fromtimestamp(ts).strftime('%Y%m%d-%H%M%S')
        job_data = {
            'orig_path': str(orig_path),
            'base_name': base_name,
            'file_type': 'out',
            'timestamp': ts_str,
            'ext': ext,
            'is_tuple': False,
            'tuple_index': idx,
            'subid': 'c'
        }
        job_id = f"single_{idx:04d}"
        jobs.append(Job(job_id=job_id, data=job_data))
    print(f"🔧 Created {len(jobs)} anonymized job(s) for single-file batch.")
    return jobs

def extract_type(filename):
    for key, mapped in TYPE_MAP.items():
        if key in filename:
            return mapped, key
    return None, None

# Patch process_file_job to use new filename format for tuples
def process_file_job_with_subid(job, run_folder: Path):
    from file_ingestion import process_file_job as orig_process_file_job
    # Patch the filename logic for tuples
    if job.data['is_tuple']:
        index_str = f"{job.data['tuple_index']:04d}"
        subid = job.data['subid']
        ts_str = job.data['timestamp']
        file_type = job.data['file_type']
        ext = job.data['ext']
        new_name = f"{index_str}-{subid}-{file_type}-{ts_str}{ext}"
        job.data['output_name'] = new_name
    return orig_process_file_job(job, run_folder)

# Utility to estimate token count
try:
    import tiktoken
    def estimate_tokens(text, model="gpt-3.5-turbo"):
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
except ImportError:
    def estimate_tokens(text, model=None):
        # Fallback: 1 token ≈ 4 chars (OpenAI heuristic)
        return max(1, len(text) // 4)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Audio Context Tool Orchestrator")
    parser.add_argument('input_dir', type=str, nargs='?', help='Input directory (required for fresh runs, optional when resuming)')
    parser.add_argument('--output-folder', type=str, help='Existing output folder to resume from (e.g., outputs/run-20250522-211044)')
    parser.add_argument('--asr_engine', type=str, choices=['parakeet', 'whisper'], default='parakeet', help='ASR engine to use for transcription (default: parakeet)')
    parser.add_argument('--llm_config', type=str, default=None, help='Path to LLM task config JSON (default: workflows/llm_tasks.json)')
    parser.add_argument('--llm-seed', type=int, default=None, help='Global seed for LLM tasks (default: random per task)')
    parser.add_argument('--call-tones', action='store_true', help='Append tones.wav to end of each call and between calls in show file')
    parser.add_argument('--call-cutter', action='store_true', help='Enable CLAP-based call segmentation for long files')
    parser.add_argument('--url', action='append', help='Download and process audio from one or more URLs')
    parser.add_argument('--mode', type=str, choices=['auto', 'single-file', 'calls'], default='auto', help='Workflow mode: auto (default), single-file, or calls')
    # Resume functionality arguments
    parser.add_argument('--resume', action='store_true', help='Enable resume functionality - skip completed stages')
    parser.add_argument('--resume-from', type=str, metavar='STAGE', help='Resume from a specific stage (skip all prior completed stages)')
    parser.add_argument('--force-rerun', type=str, metavar='STAGE', help='Force re-run a specific stage even if marked complete')
    parser.add_argument('--clear-from', type=str, metavar='STAGE', help='Clear completion status from specified stage onwards')
    parser.add_argument('--stage-status', type=str, metavar='STAGE', help='Show detailed status for a specific stage')
    parser.add_argument('--show-resume-status', action='store_true', help='Show current resume status and exit')
    parser.add_argument('--force', action='store_true', help='When used with --resume-from, deletes all outputs and state from that stage forward for a clean re-run')
    # 1. Add CLI flag for out-file processing
    parser.add_argument('--process-out-files', action='store_true', help='Enable processing of out- files as single-file inputs (lower fidelity, not recommended for main workflow)')

    # Print valid stage names in help
    from pipeline_state import get_pipeline_stages
    STAGE_LIST = get_pipeline_stages()
    parser.epilog = (
        "\nValid stage names for --resume-from, --force-rerun, --clear-from, --stage-status:\n  " + ", ".join(STAGE_LIST) +
        "\n\nExample: --resume-from diarization\n" +
        "\nUse --force with --resume-from to delete all outputs and state from that stage forward.\n" +
        "\nUse --llm-seed to set a global seed for LLM tasks (default: random per task).\n" +
        "\nUse --call-cutter to enable CLAP-based call segmentation for long files.\n" +
        "\n\nNote: By default, out- files are only used for CLAP segmentation/annotation. Use --process-out-files to process them as single-file inputs (not recommended for main workflow)."
    )

    args = parser.parse_args()

    # --- Dedicated --url first stage ---
    import os
    from pathlib import Path
    import shutil
    import time
    from audio_separation import separate_single_audio_file
    if args.url:
        # Download audio from URL(s) using yt-dlp into raw_inputs/input_from_url.wav
        run_folder = Path(args.output_folder) if args.output_folder else Path('outputs') / f'run-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        raw_inputs_dir = run_folder / 'raw_inputs'
        raw_inputs_dir.mkdir(parents=True, exist_ok=True)
        import yt_dlp
        yt_metadata_list = []
        for url in args.url:
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': str(raw_inputs_dir / 'input_from_url.%(ext)s'),  # Always use known filename
                'quiet': True,
                'no_warnings': True,
                'noplaylist': True,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'writethumbnail': False,
                'writeinfojson': True,
            }
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    yt_metadata = {
                        'source_url': url,
                        'title': info.get('title', 'downloaded'),
                        'uploader': info.get('uploader'),
                        'upload_date': info.get('upload_date'),
                        'duration': info.get('duration'),
                        'original_ext': info.get('ext'),
                        'id': info.get('id'),
                        'info_dict': info
                    }
                    yt_metadata_list.append(yt_metadata)
            except Exception as e:
                print(f"[ERROR] yt-dlp download failed for {url}: {e}")
                exit(1)
        # Save yt-dlp metadata
        import json
        yt_metadata_path = raw_inputs_dir / 'yt_metadata.json'
        with open(yt_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(yt_metadata_list, f, indent=2)
        # Find the downloaded .wav file and rename to input_from_url.wav if needed
        wav_files = list(raw_inputs_dir.glob('input_from_url.*.wav'))
        if not wav_files:
            wav_files = list(raw_inputs_dir.glob('input_from_url.wav'))
        if wav_files:
            url_wav = wav_files[0]
            canonical_wav = raw_inputs_dir / 'input_from_url.wav'
            if url_wav != canonical_wav:
                import shutil
                shutil.copy2(url_wav, canonical_wav)
            print(f"[INFO] Downloaded and prepared {canonical_wav}")
        else:
            print(f"[ERROR] No .wav file found after yt-dlp download. Aborting pipeline.")
            exit(1)
        # Proceed with dedicated single-file separation
        # Instead of separated_dir = run_folder / 'separated' / 'url_input', use separated_dir = run_folder / 'separated' / '0000'
        separated_dir = run_folder / 'separated' / '0000'
        separated_dir.mkdir(parents=True, exist_ok=True)
        url_wav = raw_inputs_dir / 'input_from_url.wav'
        if url_wav.exists():
            print(f"[INFO] Running dedicated single-file separation for {url_wav}")
            model_path = 'mel_band_roformer_vocals_fv4_gabox.ckpt'
            # Output directly to separated/0000/
            result = separate_single_audio_file(url_wav, separated_dir, model_path)
            print(f"[INFO] Separation result: {result}")
            # Find vocals.wav from output_stems
            vocals_path = None
            instrumental_path = None
            for stem in result.get('output_stems', []):
                if stem.get('stem_type') == 'vocals' and stem.get('output_path'):
                    vocals_path = Path(stem['output_path'])
                if stem.get('stem_type') == 'instrumental' and stem.get('output_path'):
                    instrumental_path = Path(stem['output_path'])
            # Move/rename stems to separated/0000/ with anonymized names
            moved_vocals = None
            moved_instrumental = None
            if vocals_path and vocals_path.exists():
                moved_vocals = separated_dir / '0000-out-vocals.wav'
                shutil.copy2(vocals_path, moved_vocals)
                print(f"[INFO] Moved vocals stem to {moved_vocals}")
            if instrumental_path and instrumental_path.exists():
                moved_instrumental = separated_dir / '0000-out-instrumental.wav'
                shutil.copy2(instrumental_path, moved_instrumental)
                print(f"[INFO] Moved instrumental stem to {moved_instrumental}")
            # Optionally remove the now-empty subfolder
            if vocals_path and vocals_path.parent != separated_dir:
                try:
                    import os
                    os.remove(vocals_path)
                    if instrumental_path:
                        os.remove(instrumental_path)
                    os.rmdir(vocals_path.parent)
                except Exception as e:
                    print(f"[DEBUG] Could not clean up subfolder: {e}")
            # Use moved_vocals as canonical input for downstream pipeline
            if moved_vocals and moved_vocals.exists():
                downstream_input_dir = run_folder / 'separated_url_input_for_pipeline'
                downstream_input_dir.mkdir(parents=True, exist_ok=True)
                canonical_input = downstream_input_dir / '0000-out-urlinput.wav'
                shutil.copy2(moved_vocals, canonical_input)
                print(f"[INFO] Copied vocals stem to {canonical_input} for downstream pipeline")
                args.input_dir = str(downstream_input_dir)
            else:
                print(f"[ERROR] vocals.wav not found after separation. Searched output_stems and got: {result.get('output_stems', [])}")
                # Log all files in separated_dir and its subfolders for debugging
                for root, dirs, files in os.walk(separated_dir):
                    print(f"[DEBUG] Files in {root}: {files}")
                exit(1)
        else:
            print(f"[ERROR] input_from_url.wav not found in raw_inputs. Aborting pipeline.")
            exit(1)
        print(f"[DEBUG] After --url block: args.input_dir = {args.input_dir}")
        print(f"[DEBUG] After --url block: args.output_folder = {args.output_folder}")

    # --- Main orchestrator creation and pipeline run logic ---
    print(f"[DEBUG] Entering main pipeline logic. args.input_dir = {args.input_dir}")
    if args.output_folder:
        run_folder = Path(args.output_folder)
        print(f"[DEBUG] Using run_folder: {run_folder}")
        if not run_folder.exists():
            print(f"❌ Output folder not found: {run_folder}")
            # ... existing code ...
        print(f"🔄 Resuming from existing folder: {run_folder}")
        # ... existing code ...
        print(f"[DEBUG] Creating orchestrator for existing folder: {run_folder}")
        orchestrator = PipelineOrchestrator(run_folder, args.asr_engine, args.llm_config)
        # ... existing code ...
    else:
        # Starting fresh run
        if not args.input_dir:
            print("❌ input_dir is required when starting a fresh run")
            # ... existing code ...
        # ... existing code ...
        input_dir = Path(args.input_dir)
        run_ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        run_folder = Path('outputs') / f'run-{run_ts}'
        print(f"🆕 Starting fresh run: {run_folder}")
        print(f"[DEBUG] Creating orchestrator for fresh run: {run_folder}, input_dir: {input_dir}")
        orchestrator = PipelineOrchestrator(run_folder, args.asr_engine, args.llm_config)
        jobs = create_jobs_from_input(input_dir)
        print(f"[DEBUG] Created {len(jobs)} jobs for orchestrator")
        for job in jobs:
            orchestrator.add_job(job)
        # ... existing code ...
    resume_mode = args.resume or args.resume_from or args.force_rerun or args.output_folder
    resume_from_stage = args.resume_from
    print(f"[DEBUG] About to run pipeline. resume_mode={resume_mode}, resume_from_stage={resume_from_stage}")
    if resume_mode:
        print(f"[DEBUG] Calling orchestrator.run_with_resume()")
        orchestrator.run_with_resume(call_tones=args.call_tones, resume=True, resume_from=resume_from_stage)
        print(f"[DEBUG] Finished orchestrator.run_with_resume()")
    else:
        print(f"[DEBUG] Calling orchestrator.run()")
        orchestrator.run(mode=args.mode, call_cutter=args.call_cutter, call_tones=args.call_tones)
        print(f"[DEBUG] Finished orchestrator.run()")