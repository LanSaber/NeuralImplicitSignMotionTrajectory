#!/usr/bin/env python
import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_DATA_DIR = Path("/media/cvpr/haomian/data/SOKE_FLOW/phoenix_upper_smplx")
BLANK_TOKEN = "<blank>"


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def gloss_tokens(value):
    return str(value or "").split()


def build_vocab(data_dir, train_split="train"):
    manifest_path = Path(data_dir) / "meta" / f"manifest_{train_split}.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing train manifest: {manifest_path}")

    counts = Counter()
    num_sequences = 0
    num_tokens = 0
    for row in read_jsonl(manifest_path):
        tokens = gloss_tokens(row.get("gloss", ""))
        if not tokens:
            continue
        counts.update(tokens)
        num_sequences += 1
        num_tokens += len(tokens)

    glosses = sorted(counts)
    gloss_to_id = {gloss: idx + 1 for idx, gloss in enumerate(glosses)}
    id_to_gloss = [BLANK_TOKEN] + glosses
    return {
        "blank_id": 0,
        "blank_token": BLANK_TOKEN,
        "gloss_to_id": gloss_to_id,
        "id_to_gloss": id_to_gloss,
        "vocab_size_without_blank": len(glosses),
        "vocab_size_with_blank": len(id_to_gloss),
        "train_split": train_split,
        "num_train_sequences": num_sequences,
        "num_train_tokens": num_tokens,
        "gloss_counts": dict(sorted(counts.items())),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a Phoenix train-split gloss vocabulary for CTC alignment."
    )
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--train_split", default="train")
    parser.add_argument(
        "--out_path",
        type=Path,
        default=None,
        help="Defaults to DATA_DIR/meta/gloss_vocab.json.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_path = args.out_path or args.data_dir / "meta" / "gloss_vocab.json"
    vocab = build_vocab(args.data_dir, args.train_split)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(vocab, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"Wrote {vocab['vocab_size_without_blank']} glosses "
        f"({vocab['vocab_size_with_blank']} with blank) to {out_path}"
    )


if __name__ == "__main__":
    main()
