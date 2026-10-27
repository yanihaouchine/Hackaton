import sys
from pathlib import Path

# l'oracle Python (tests/reference_engine.py) est importé directement par les tests
sys.path.insert(0, str(Path(__file__).parent))
