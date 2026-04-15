import argparse
import json
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm


def clean_caption(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return f"startseq {text} endseq"


def load_and_clean_captions(captions_path: Path) -> pd.DataFrame:
    df = pd.read_csv(captions_path)
    if "image" not in df.columns or "caption" not in df.columns:
        raise ValueError("captions.txt must contain 'image' and 'caption' columns.")
    df["cleaned_caption"] = df["caption"].astype(str).apply(clean_caption)
    return df


def build_tokenizer(captions: pd.Series, num_words: int) -> Tokenizer:
    tokenizer = Tokenizer(num_words=num_words, oov_token="<unk>")
    tokenizer.fit_on_texts(captions.tolist())
    return tokenizer


def extract_features(images_dir: Path, image_names: list[str], batch_size: int = 32) -> dict[str, np.ndarray]:
    model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    features: dict[str, np.ndarray] = {}

    for i in tqdm(range(0, len(image_names), batch_size), desc="Extracting CNN features"):
        batch_names = image_names[i : i + batch_size]
        batch_images = []
        valid_names = []

        for name in batch_names:
            img_path = images_dir / name
            if not img_path.exists():
                continue
            img = image.load_img(img_path, target_size=(224, 224))
            arr = image.img_to_array(img)
            batch_images.append(arr)
            valid_names.append(name)

        if not batch_images:
            continue

        arr_batch = np.array(batch_images, dtype="float32")
        arr_batch = preprocess_input(arr_batch)
        preds = model.predict(arr_batch, verbose=0)

        for name, feat in zip(valid_names, preds):
            features[name] = feat.astype("float32")

    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Flickr8k captions and image features.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=r"e:\dl\archive (1)",
        help="Directory containing captions.txt and Images/.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"e:\dl\image_captioning\artifacts",
        help="Directory to save tokenizer, metadata, and image features.",
    )
    parser.add_argument("--max_vocab", type=int, default=5000, help="Maximum vocabulary size.")
    parser.add_argument("--test_size", type=float, default=0.1, help="Validation split at image level.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    captions_path = dataset_dir / "captions.txt"
    images_dir = dataset_dir / "Images"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not captions_path.exists():
        raise FileNotFoundError(f"Missing captions file: {captions_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing Images directory: {images_dir}")

    print("Loading and cleaning captions...")
    df = load_and_clean_captions(captions_path)

    unique_images = sorted(df["image"].unique().tolist())
    train_images, val_images = train_test_split(
        unique_images,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    df["split"] = df["image"].apply(lambda x: "val" if x in set(val_images) else "train")

    tokenizer = build_tokenizer(df["cleaned_caption"], num_words=args.max_vocab)
    max_length = int(df["cleaned_caption"].str.split().apply(len).max())
    vocab_size = min(args.max_vocab, len(tokenizer.word_index) + 1)

    print(f"Total captions: {len(df)}")
    print(f"Unique images: {len(unique_images)}")
    print(f"Train images: {len(train_images)} | Val images: {len(val_images)}")
    print(f"Tokenizer vocab size: {vocab_size}")
    print(f"Maximum caption length: {max_length}")

    cleaned_csv_path = output_dir / "cleaned_captions.csv"
    df.to_csv(cleaned_csv_path, index=False)

    tokenizer_path = output_dir / "tokenizer.pkl"
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)

    metadata = {
        "dataset_dir": str(dataset_dir),
        "images_dir": str(images_dir),
        "max_length": max_length,
        "vocab_size": vocab_size,
        "max_vocab": args.max_vocab,
        "num_unique_images": len(unique_images),
        "num_captions": len(df),
        "train_images": len(train_images),
        "val_images": len(val_images),
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Extracting image features with ResNet50...")
    features = extract_features(images_dir, unique_images, batch_size=32)
    features_path = output_dir / "image_features.pkl"
    with open(features_path, "wb") as f:
        pickle.dump(features, f)

    print(f"Saved cleaned captions to: {cleaned_csv_path}")
    print(f"Saved tokenizer to: {tokenizer_path}")
    print(f"Saved metadata to: {metadata_path}")
    print(f"Saved image features to: {features_path}")


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
