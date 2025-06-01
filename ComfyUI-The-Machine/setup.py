from setuptools import setup, find_packages
import os

# Read long description from README.md
with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='ComfyUI-The-Machine',
    version='0.1.0',
    description='A modular, privacy-first audio processing pipeline for ComfyUI as custom nodes.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='The-Machine Team',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'torch>=2.0.0',
        'torchaudio>=2.0.0',
        'tqdm',
        'numpy',
        'soundfile',
        'mutagen',
        'pydub',
        'scipy',
        'colorama',
        'pyyaml',
        'requests',
        'transformers>=4.30.0',
        'datasets',
        'huggingface_hub',
        'demucs',
        'spleeter',
        'pyloudnorm',
        'pyannote.audio',
        'comfyui',
        'ffmpeg-python',
        'yt_dlp',
        'openai-whisper',
        'nemo_toolkit[asr]>=1.22.0',
    ],
    include_package_data=True,
    zip_safe=False,
    # If ComfyUI requires entry_points for node discovery, add here:
    # entry_points={
    #     'comfyui.nodes': [
    #         'the_machine = ComfyUI-The-Machine:register_nodes',
    #     ],
    # },
    # TODO: If ComfyUI requires a specific registration function, implement and document it.
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
) 