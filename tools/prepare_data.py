"""Download the original LAFAN1/G1 assets, or copy them from an existing data directory."""

import argparse
import json
from pathlib import Path
import shutil
import time
import urllib.request
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[1]
BASE = "https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset"
API = "https://huggingface.co/api/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/main"


def download(remote: str, dest: Path):
    if dest.is_file() and dest.stat().st_size:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(f"{BASE}/resolve/main/{remote}", timeout=120) as response, temp.open("wb") as output:
                shutil.copyfileobj(response, output)
            temp.replace(dest)
            print(dest.relative_to(REPO), flush=True)
            return
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="existing data directory to copy (no network)")
    args = parser.parse_args()
    data = REPO / "data"
    robot = data / "g1_description"
    if args.source:
        for directory in ("lafan1_g1", "g1_description"):
            shutil.copytree(args.source / directory, data / directory, dirs_exist_ok=True)
    else:
        with urllib.request.urlopen(f"{API}/g1?limit=1000", timeout=60) as response:
            files = json.load(response)
        for entry in files:
            if entry["type"] == "file" and entry["path"].endswith(".csv"):
                download(entry["path"], data / "lafan1_g1" / Path(entry["path"]).name)
        download("robot_description/g1/g1_29dof_rev_1_0.urdf", robot / "g1_29dof_rev_1_0.urdf")
        tree = ET.parse(robot / "g1_29dof_rev_1_0.urdf")
        meshes = {node.attrib["filename"] for node in tree.findall(".//mesh")}
        for mesh in sorted(meshes):
            download(f"robot_description/g1/{mesh}", robot / mesh)
    tree = ET.parse(robot / "g1_29dof_rev_1_0.urdf")
    for node in tree.findall(".//mesh"):
        path = robot / node.attrib["filename"]
        if not path.is_file():
            raise FileNotFoundError(path)
    clips = list((data / "lafan1_g1").glob("*.csv"))
    if not clips:
        raise FileNotFoundError("No motion CSVs available")
    print(f"Prepared {len(clips)} clips and G1 meshes. Data licenses: LAFAN1 CC BY-NC-ND 4.0; G1 BSD-3-Clause.")


if __name__ == "__main__":
    main()
