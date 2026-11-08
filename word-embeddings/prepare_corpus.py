"""One-off: stream TinyStories from HuggingFace and save a plain-text sample."""

from pathlib import Path

from datasets import load_dataset

from word_embeddings import tokenize

TARGET_TOKENS = 20_000_000
OUTPUT = Path("data/tinystories_20m.txt")


def main() -> None:
    OUTPUT.parent.mkdir(exist_ok=True)
    stream = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    written = 0
    stories = 0
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for story in stream:
            text = story["text"].replace("\n", " ").strip()
            handle.write(text + "\n")
            written += len(tokenize(text))
            stories += 1
            if stories % 20_000 == 0:
                print(f"stories={stories:,} tokens={written:,}", flush=True)
            if written >= TARGET_TOKENS:
                break

    print(f"done: {stories:,} stories, {written:,} tokens -> {OUTPUT}")


if __name__ == "__main__":
    main()
