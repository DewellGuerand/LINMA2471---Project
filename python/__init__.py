# Make subpackages easily importable with relative imports
# This allows: from models import SmoothMarkowitzModel
# when running from within the python/ directory

import sys
from pathlib import Path

# Add the python/ directory to the path so subpackages can find each other
_python_dir = Path(__file__).parent
if str(_python_dir) not in sys.path:
    sys.path.insert(0, str(_python_dir))
