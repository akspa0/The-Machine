# ComfyUI-The-Machine: Node registration
# See ComfyUI custom node API documentation for details.

from .node_input_ingestion import RawInputIngestionNode
from .node_tuple_assembler import TupleAssemblerNode
from .node_separation import SeparationNode
from .node_normalization import NormalizationNode
from .node_clap_annotation import CLAPAnnotationNode
from .node_diarization import DiarizationNode
from .node_transcription import TranscriptionNode
from .node_soundbite_generation import SoundbiteGenerationNode
from .node_llm_task import LLMTaskNode
from .node_remixing import RemixingNode
from .node_show_output import ShowOutputNode
from .node_export_database import ExportDatabaseNode

NODE_CLASSES = [
    RawInputIngestionNode,
    TupleAssemblerNode,
    SeparationNode,
    NormalizationNode,
    CLAPAnnotationNode,
    DiarizationNode,
    TranscriptionNode,
    SoundbiteGenerationNode,
    LLMTaskNode,
    RemixingNode,
    ShowOutputNode,
    ExportDatabaseNode,
]

def register_nodes():
    # This function will be called by ComfyUI to register all nodes
    for node_class in NODE_CLASSES:
        # Placeholder: actual registration logic as per ComfyUI API
        pass 