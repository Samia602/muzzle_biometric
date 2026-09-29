import json
import numpy as np
import torch

from config import REGISTRY_FILE
from src.pipeline import embedding_hash


class Registry:
    """Tiny local stand-in for 'blockchain + IPFS' while you build phase 1."""

    def __init__(self, path=REGISTRY_FILE):
        self.path = path

        if path.exists():
            self.data = json.loads(path.read_text())
        else:
            self.data = {}

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, indent=2)
        )

    def register(self, animal_id, embeddings):

        if not embeddings:
            raise ValueError("No embeddings provided.")

        # Convert embeddings to NumPy
        arrays = []

        for emb in embeddings:
            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu().numpy()

            arrays.append(np.asarray(emb, dtype=np.float32))

        # Average multiple images of the same cow
        m = np.mean(
            np.stack(arrays),
            axis=0
        )

        # Normalize averaged embedding
        norm = np.linalg.norm(m)

        if norm == 0:
            raise ValueError("Embedding has zero norm.")

        m = m / norm

        # Convert back to PyTorch tensor for embedding_hash()
        embedding_tensor = torch.tensor(
            m,
            dtype=torch.float32
        )

        biometric_hash = embedding_hash(
            embedding_tensor
        )

        self.data[animal_id] = {
            "embedding": m.tolist(),
            "hash": biometric_hash,
            "n_images": len(embeddings)
        }

        self._save()

        return biometric_hash

    def identify(self, emb, top_k=3):

        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().numpy()

        emb = np.asarray(
            emb,
            dtype=np.float32
        )

        # Normalize query embedding
        norm = np.linalg.norm(emb)

        if norm == 0:
            raise ValueError("Query embedding has zero norm.")

        emb = emb / norm

        results = []

        for animal_id, value in self.data.items():

            stored = np.asarray(
                value["embedding"],
                dtype=np.float32
            )

            # Stored embeddings are already normalized
            similarity = float(
                np.dot(emb, stored)
            )

            results.append(
                (animal_id, similarity)
            )

        return sorted(
            results,
            key=lambda x: -x[1]
        )[:top_k]

    def clear(self):

        self.data = {}
        self._save()