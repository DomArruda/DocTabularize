"""
Example runner for Document Refinery.
Place your PDF in input/ and configure pipeline_config.toml
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.doctabularize.pipeline import run
from src.doctabularize.config import Config

if __name__ == "__main__":
    # Optional: override config path
    # cfg = Config("custom_config.toml")
    run()