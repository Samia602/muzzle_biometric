import json
import numpy as np
from config import REGISTRY_FILE
from src.pipeline import embedding_hash


class Registry:
    """Tiny local stand-in for 'blockchain + IPFS' while you build phase 1."""

    def __init__(self, path=REGISTRY_FILE):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {}

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data))

    def register(self, animal_id, embeddings):
        m = np.mean(np.stack(embeddings), axis=0)
        m /= np.linalg.norm(m)
        self.data[animal_id] = {"embedding": m.tolist(), "hash": embedding_hash(m), "n_images": len(embeddings)}
        self._save()
        return self.data[animal_id]["hash"]

    def identify(self, emb, top_k=3):
        res = [(k, float(np.dot(emb, np.array(v["embedding"])))) for k, v in self.data.items()]
        return sorted(res, key=lambda x: -x[1])[:top_k]

    def clear(self):
        self.data = {}
        self._save()
