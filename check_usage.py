import sys
import os
sys.path.append("scripts")
from prepare_data import install_convert_genome
import subprocess

try:
    install_convert_genome()
    subprocess.call(["convert_genome", "--help"])
except Exception as e:
    print(e)
