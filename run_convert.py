import sys
from pathlib import Path
sys.path.insert(0, str(Path("scripts").absolute()))
import prepare_data
prepare_data.install_convert_genome()
