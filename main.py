"""main.py - Main entry point for the application"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from gui.app import main

if __name__ == "__main__":
    main()