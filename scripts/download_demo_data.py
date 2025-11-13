"""Example: download a small public text file with a progress bar."""

from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.download_utils import download_file


def main() -> None:
    # Tiny Shakespeare (~1 MB): great for quick progress bar demo
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    dest = Path("data/demo/tinyshakespeare.txt")
    download_file(url, dest, show_progress=True)
    print(f"Downloaded to {dest}")


if __name__ == "__main__":
    main()

