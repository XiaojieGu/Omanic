import os
import shutil
from huggingface_hub import hf_hub_download

REPO_ID = "li-lab/Omanic"
FILES = ["OmanicSynth.jsonl", "OmanicBench.jsonl"]


def main() -> None:
    output_dir = os.path.dirname(os.path.abspath(__file__))
    for filename in FILES:
        cached_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=filename,
        )
        dst_path = os.path.join(output_dir, filename)
        shutil.copy2(cached_path, dst_path)
        print(f"downloaded {filename} -> {dst_path}")


if __name__ == "__main__":
    main()
