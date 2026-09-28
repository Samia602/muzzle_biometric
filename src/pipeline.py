import hashlib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from config import YOLO_WEIGHTS, ARC_WEIGHTS
from src.models import EmbeddingNet
from src.dataset import MEAN, STD


def embedding_hash(emb: np.ndarray) -> str:
    """SHA-256 of the int8-quantised embedding (this is what you anchor on-chain).
    NOTE: exact-hash equality is NOT used for matching; matching = cosine similarity."""
    q = np.round(np.asarray(emb) * 127).astype(np.int8).tobytes()
    return hashlib.sha256(q).hexdigest()


class MuzzlePipeline:
    def __init__(self, yolo_weights=YOLO_WEIGHTS, arc_weights=ARC_WEIGHTS, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = None
        if yolo_weights and yolo_weights.exists():
            from ultralytics import YOLO
            self.detector = YOLO(str(yolo_weights))
        ck = torch.load(arc_weights, map_location=self.device)
        self.threshold = float(ck.get("threshold", 0.5))
        self.size = ck["img_size"]
        self.net = EmbeddingNet(ck["arch"], ck["emb_dim"], pretrained=False)
        self.net.load_state_dict(ck["model"])
        self.net.to(self.device).eval()
        self.tf = T.Compose([T.Resize((self.size, self.size)), T.ToTensor(), T.Normalize(MEAN, STD)])

    def detect(self, img: Image.Image, conf=0.25, margin=0.08):
        """Returns (crop, box_xyxy or None, det_conf). Falls back to full image."""
        if self.detector is None:
            return img, None, 0.0
        r = self.detector.predict(img, conf=conf, verbose=False)[0]
        if len(r.boxes) == 0:
            return img, None, 0.0
        i = int(r.boxes.conf.argmax())
        x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
        w, h = x2 - x1, y2 - y1
        W, H = img.size
        box = (max(0, x1 - margin * w), max(0, y1 - margin * h), min(W, x2 + margin * w), min(H, y2 + margin * h))
        return img.crop(tuple(int(v) for v in box)), box, float(r.boxes.conf[i])

    @torch.no_grad()
    def embed(self, crop: Image.Image) -> np.ndarray:
        x = self.tf(crop.convert("RGB")).unsqueeze(0).to(self.device)
        return self.net(x)[0].cpu().numpy()

    def process(self, img: Image.Image, use_detector=True):
        img = img.convert("RGB")
        crop, box, conf = self.detect(img) if use_detector else (img, None, 0.0)
        emb = self.embed(crop)
        return {"crop": crop, "box": box, "det_conf": conf, "embedding": emb, "hash": embedding_hash(emb)}

    @staticmethod
    def cosine(a, b) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
