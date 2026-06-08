"""
Document Intelligence Refinery
Target-score-aware extraction + per-row confidence metadata
"""

from .config import Config
from .models import Chunk, DiscoveredSchema, ExtractedPage
from .ingest import ingest, embed
from .cluster import clusterize, medoids
from .schema import discover_schema
from .verify import Verifier
from .extract import extract_page
from .memory import SchemaMemory
from .pipeline import run

__version__ = "2.3.0"
__all__ = [
    "Config", "Chunk", "DiscoveredSchema", "ExtractedPage",
    "ingest", "embed", "clusterize", "medoids",
    "discover_schema", "Verifier", "extract_page",
    "SchemaMemory", "run"
]