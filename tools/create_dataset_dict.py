import argparse
import json


def main(json_path):
    dataset_dict = {
        "splits": ["train", "test"]
    }

    with open(json_path, "w") as f:
        json.dump(dataset_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', help="json file path to save dataset dict")
    args = parser.parse_args()

    main(json_path=args.json_path)
