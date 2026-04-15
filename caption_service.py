import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.preprocessing.sequence import pad_sequences


class CaptionService:
    def __init__(self, project_dir: str | Path):
        self.project_dir = Path(project_dir)
        self.artifacts_dir = self.project_dir / "artifacts"
        self.checkpoints_dir = self.project_dir / "checkpoints"
        self._feature_extractor = None
        self._custom_model = None
        self._tokenizer = None
        self._metadata = None
        self._blip_processor = None
        self._blip_model = None

    def _load_feature_extractor(self):
        if self._feature_extractor is None:
            self._feature_extractor = ResNet50(weights="imagenet", include_top=False, pooling="avg")
        return self._feature_extractor

    def _load_tokenizer_and_metadata(self):
        if self._tokenizer is None or self._metadata is None:
            tokenizer_path = self.artifacts_dir / "tokenizer.pkl"
            metadata_path = self.artifacts_dir / "metadata.json"
            if not tokenizer_path.exists() or not metadata_path.exists():
                return None, None
            with open(tokenizer_path, "rb") as f:
                self._tokenizer = pickle.load(f)
            with open(metadata_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)
        return self._tokenizer, self._metadata

    def _load_custom_model(self):
        if self._custom_model is None:
            model_path_h5 = self.checkpoints_dir / "model.h5"
            model_path_keras = self.checkpoints_dir / "model.keras"
            model_path = model_path_h5 if model_path_h5.exists() else model_path_keras
            if not model_path.exists():
                return None
            self._custom_model = load_model(model_path)
        return self._custom_model

    def _load_blip(self):
        if self._blip_model is None or self._blip_processor is None:
            import torch
            from transformers import BlipForConditionalGeneration, BlipProcessor

            model_name = "Salesforce/blip-image-captioning-base"
            self._blip_processor = BlipProcessor.from_pretrained(model_name)
            self._blip_model = BlipForConditionalGeneration.from_pretrained(model_name)
            self._blip_model.eval()
            self._blip_model.to("cuda" if torch.cuda.is_available() else "cpu")
        return self._blip_processor, self._blip_model

    def _extract_feature_from_pil(self, pil_img: Image.Image) -> np.ndarray:
        model = self._load_feature_extractor()
        img = pil_img.convert("RGB").resize((224, 224))
        arr = keras_image.img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)
        feat = model.predict(arr, verbose=0)
        return feat

    @staticmethod
    def _generate_caption_greedy(model, tokenizer, photo_feat: np.ndarray, max_length: int) -> str:
        in_text = "startseq"
        for _ in range(max_length):
            seq = tokenizer.texts_to_sequences([in_text])[0]
            seq = pad_sequences([seq], maxlen=max_length)
            yhat = model.predict([photo_feat, seq], verbose=0)
            yhat_idx = int(np.argmax(yhat))
            word = tokenizer.index_word.get(yhat_idx)
            if word is None:
                break
            in_text += " " + word
            if word == "endseq":
                break
        caption = in_text.replace("startseq", "").replace("endseq", "").strip()
        return caption if caption else "Could not generate a caption."

    def custom_caption(self, pil_img: Image.Image) -> Optional[str]:
        tokenizer, metadata = self._load_tokenizer_and_metadata()
        model = self._load_custom_model()
        if tokenizer is None or metadata is None or model is None:
            return None
        max_length = int(metadata["max_length"])
        feat = self._extract_feature_from_pil(pil_img)
        return self._generate_caption_greedy(model, tokenizer, feat, max_length)

    def blip_caption(self, pil_img: Image.Image) -> str:
        import torch

        processor, model = self._load_blip()
        image = pil_img.convert("RGB")
        device = next(model.parameters()).device
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=30, num_beams=5)
        caption = processor.decode(output_ids[0], skip_special_tokens=True).strip()
        return caption if caption else "No caption generated."

    def caption(self, pil_img: Image.Image, mode: str = "blip") -> str:
        # User requested BLIP-only inference path for consistent behavior.
        return self.blip_caption(pil_img)
