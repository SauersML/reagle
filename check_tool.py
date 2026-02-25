import os
import sys
import platform
import json
import tarfile
import zipfile
import shutil
import tempfile
import stat
from urllib.request import Request, urlopen
from pathlib import Path

def _github_json(url):
    req = Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "test"})
    with urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def _platform_asset_tokens():
    sysname = platform.system().lower()
    machine = platform.machine().lower()
    os_tokens = []
    arch_tokens = []
    if sysname.startswith("linux"): os_tokens.extend(["linux"])
    if machine in ("x86_64", "amd64"): arch_tokens.extend(["x86_64", "amd64", "x64"])
    return os_tokens, arch_tokens

def _select_asset(assets):
    os_tokens, arch_tokens = _platform_asset_tokens()
    for asset in assets:
        name = asset.get("name", "").lower()
        if "convert_genome" in name and any(t in name for t in os_tokens) and any(t in name for t in arch_tokens):
            return asset
    return None

def download_and_help():
    print("Getting release info...")
    release = _github_json("https://api.github.com/repos/SauersML/convert_genome/releases/latest")
    asset = _select_asset(release.get("assets", []))
    if not asset:
        print("No asset found")
        return

    url = asset["browser_download_url"]
    name = asset["name"]
    print(f"Downloading {name} from {url}")

    with urlopen(url) as resp, open(name, "wb") as out:
        shutil.copyfileobj(resp, out)

    print("Extracting...")
    if name.endswith(".tar.gz"):
        with tarfile.open(name, "r:*") as tf:
            tf.extractall("extracted")

    bin_path = None
    for root, dirs, files in os.walk("extracted"):
        for f in files:
            if "convert_genome" in f:
                bin_path = os.path.join(root, f)
                break

    if bin_path:
        os.chmod(bin_path, 0o755)
        print(f"Running {bin_path} --help")
        os.system(f"{bin_path} --help")

if __name__ == "__main__":
    download_and_help()
