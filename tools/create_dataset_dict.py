import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ates.io import write_json


def main(json_path):
    dataset_dict = {
        "splits": ["train", "test"]
    }
    write_json(json_path, dataset_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', help="json file path to save dataset dict")
    args = parser.parse_args()

    main(json_path=args.json_path)
