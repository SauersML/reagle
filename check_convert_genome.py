
import sys
import os
sys.path.append("scripts")
import prepare_data
import subprocess

try:
    prepare_data.install_convert_genome()
    subprocess.check_call(["convert_genome", "--help"])
except Exception as e:
    print(e)
