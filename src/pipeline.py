import hashlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from src.models import EmbeddingNet
def embedding_hash(embedding):
    embedding_bytes = (
        embedding.detach()
        .cpu()
        .numpy()
        .astype(np.float32)
        .tobytes()
    )

    return hashlib.sha256(
        embedding_bytes
    ).hexdigest()

class MuzzlePipeline:

    def __init__(
        self,
        yolo_path="models/yolo_muzzle.pt",
        arcface_path="models/arcface_muzzle.pt",
        threshold=0.278
    ):

        self.device = "cpu"
        self.threshold = threshold

        # YOLO muzzle detector
        self.yolo = YOLO(yolo_path)
        self.detector = self.yolo

        # ArcFace / ResNet18 embedding model
        checkpoint = torch.load(
            arcface_path,
            map_location="cpu"
        )

        self.embedding_model = EmbeddingNet(
            embedding_dim=512
        )

        self.embedding_model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.embedding_model.eval()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406
                ],
                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            )
        ])

    # ---------------------------------------------------------
    # MUZZLE DETECTION
    # ---------------------------------------------------------

    def detect_muzzle(self, image):

        results = self.yolo.predict(
            source=image,
            device="cpu",
            imgsz=416,
            conf=0.25,
            verbose=False
        )

        result = results[0]

        if result.boxes is None:
            return None

        if len(result.boxes) == 0:
            return None

        confidences = (
            result.boxes.conf
            .cpu()
            .numpy()
        )

        best_index = int(
            np.argmax(confidences)
        )

        box = (
            result.boxes.xyxy[best_index]
            .cpu()
            .numpy()
            .astype(int)
        )

        x1, y1, x2, y2 = box

        image_array = np.array(image)

        h, w = image_array.shape[:2]

        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        crop = image_array[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        return (
            Image.fromarray(crop),
            box,
            float(confidences[best_index])
        )

    # ---------------------------------------------------------
    # EMBEDDING
    # ---------------------------------------------------------

    def embed(self, image):

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = image.convert("RGB")

        tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            embedding = self.embedding_model(tensor)

        return embedding[0].cpu()

    # ---------------------------------------------------------
    # COMPLETE PIPELINE
    # ---------------------------------------------------------

    def process(self, image, use_detector=True):

        image = image.convert("RGB")

        box = None
        crop = image
        det_conf = 0.0

        if use_detector:

            detection = self.detect_muzzle(image)

            if detection is not None:

                crop, box, det_conf = detection

        embedding = self.embed(crop)

        # SHA-256 hash of embedding
        biometric_hash = embedding_hash(embedding)

        return {
            "embedding": embedding,
            "box": box,
            "crop": crop,
            "det_conf": det_conf,
            "hash": biometric_hash
        }

    # ---------------------------------------------------------
    # COSINE SIMILARITY
    # ---------------------------------------------------------

    def cosine(self, embedding1, embedding2):

        return F.cosine_similarity(
            embedding1.unsqueeze(0),
            embedding2.unsqueeze(0)
        ).item()

    # ---------------------------------------------------------
    # COMPARE
    # ---------------------------------------------------------

    def compare(self, embedding1, embedding2):

        score = self.cosine(
            embedding1,
            embedding2
        )

        same = score >= self.threshold

        return same, score

