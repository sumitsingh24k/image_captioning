import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Add, Dense, Dropout, Embedding, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence


class CaptionDataGenerator(Sequence):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        features: dict[str, np.ndarray],
        tokenizer,
        max_length: int,
        vocab_size: int,
        batch_size: int = 64,
        shuffle: bool = True,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.features = features
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        self.samples = self._build_samples()
        self.on_epoch_end()

    def _build_samples(self) -> list[tuple[str, list[int], int]]:
        samples: list[tuple[str, list[int], int]] = []
        for _, row in self.df.iterrows():
            image_name = row["image"]
            if image_name not in self.features:
                continue
            seq = self.tokenizer.texts_to_sequences([row["cleaned_caption"]])[0]
            for i in range(1, len(seq)):
                in_seq = seq[:i]
                out_seq = seq[i]
                if out_seq < self.vocab_size:
                    samples.append((image_name, in_seq, out_seq))
        return samples

    def __len__(self) -> int:
        return math.ceil(len(self.samples) / self.batch_size)

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.samples)

    def __getitem__(self, index: int):
        batch = self.samples[index * self.batch_size : (index + 1) * self.batch_size]

        X_img, X_seq, y = [], [], []
        for image_name, in_seq, out_seq in batch:
            X_img.append(self.features[image_name])
            X_seq.append(pad_sequences([in_seq], maxlen=self.max_length)[0])
            y.append(out_seq)

        return (np.array(X_img), np.array(X_seq)), np.array(y)


def build_model(vocab_size: int, max_length: int) -> Model:
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation="relu")(fe1)

    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = Add()([fe2, se3])
    decoder2 = Dense(256, activation="relu")(decoder1)
    outputs = Dense(vocab_size, activation="softmax")(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN-LSTM image captioning model.")
    parser.add_argument(
        "--artifacts_dir",
        type=str,
        default=r"e:\dl\image_captioning\artifacts",
        help="Directory containing cleaned_captions.csv, tokenizer.pkl, metadata.json, image_features.pkl",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size.")
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default=r"e:\dl\image_captioning\checkpoints",
        help="Directory for saving trained models.",
    )
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    checkpoints_dir = Path(args.checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    with open(artifacts_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(artifacts_dir / "tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open(artifacts_dir / "image_features.pkl", "rb") as f:
        image_features = pickle.load(f)

    captions_df = pd.read_csv(artifacts_dir / "cleaned_captions.csv")
    train_df = captions_df[captions_df["split"] == "train"].copy()
    val_df = captions_df[captions_df["split"] == "val"].copy()

    vocab_size = int(metadata["vocab_size"])
    max_length = int(metadata["max_length"])

    train_gen = CaptionDataGenerator(
        dataframe=train_df,
        features=image_features,
        tokenizer=tokenizer,
        max_length=max_length,
        vocab_size=vocab_size,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_gen = CaptionDataGenerator(
        dataframe=val_df,
        features=image_features,
        tokenizer=tokenizer,
        max_length=max_length,
        vocab_size=vocab_size,
        batch_size=args.batch_size,
        shuffle=False,
    )

    print(f"Training samples: {len(train_gen.samples)}")
    print(f"Validation samples: {len(val_gen.samples)}")
    print(f"Vocab size: {vocab_size}, Max length: {max_length}")

    model = build_model(vocab_size=vocab_size, max_length=max_length)
    model.summary()

    best_path = checkpoints_dir / "best_model.keras"
    callbacks = [
        ModelCheckpoint(
            filepath=str(best_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    keras_path = checkpoints_dir / "model.keras"
    h5_path = checkpoints_dir / "model.h5"
    model.save(keras_path)
    model.save(h5_path)

    print(f"Saved model: {keras_path}")
    print(f"Saved model: {h5_path}")


if __name__ == "__main__":
    main()
