import pytest

from ComfyUI-The-Machine.node_input_ingestion import RawInputIngestionNode
from ComfyUI-The-Machine.node_tuple_assembler import TupleAssemblerNode
from ComfyUI-The-Machine.node_separation import SeparationNode
from ComfyUI-The-Machine.node_normalization import NormalizationNode
from ComfyUI-The-Machine.node_clap_annotation import CLAPAnnotationNode
from ComfyUI-The-Machine.node_diarization import DiarizationNode
from ComfyUI-The-Machine.node_transcription import TranscriptionNode
from ComfyUI-The-Machine.node_soundbite_generation import SoundbiteGenerationNode
from ComfyUI-The-Machine.node_llm_task import LLMTaskNode
from ComfyUI-The-Machine.node_remixing import RemixingNode
from ComfyUI-The-Machine.node_show_output import ShowOutputNode
from ComfyUI-The-Machine.node_export_database import ExportDatabaseNode

def test_node_instantiation():
    # Test that all node classes can be instantiated
    RawInputIngestionNode()
    TupleAssemblerNode()
    SeparationNode()
    NormalizationNode()
    CLAPAnnotationNode()
    DiarizationNode()
    TranscriptionNode()
    SoundbiteGenerationNode()
    LLMTaskNode()
    RemixingNode()
    ShowOutputNode()
    ExportDatabaseNode() 