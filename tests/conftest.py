"""pytest configuration for rp2 tests."""
import sys
from pathlib import Path

# Ensure ai-analysis is in the path
project_root = Path(__file__).parent.parent
ai_analysis_path = str(project_root / "ai-analysis")

if ai_analysis_path not in sys.path:
    sys.path.insert(0, ai_analysis_path)
