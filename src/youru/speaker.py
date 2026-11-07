"""Speaker embeddings (sherpa-onnx) and the owner voice profile."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE


def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


class Embedder:
    def __init__(self, model_path: str | Path, threads: int = 2):
        import sherpa_onnx

        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model_path), num_threads=threads,
                                                          provider="cpu")
        if not cfg.validate():
            raise ValueError(f"bad speaker model: {model_path}")
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        self.model_name = Path(model_path).stem

    def embed(self, audio: np.ndarray) -> np.ndarray:
        s = self.extractor.create_stream()
        s.accept_waveform(sample_rate=SAMPLE_RATE, waveform=np.ascontiguousarray(audio, dtype=np.float32))
        s.input_finished()
        return normalize(np.array(self.extractor.compute(s), dtype=np.float32))


class OwnerProfile:
    """Enrollment embeddings (fixed) plus a rolling bank of adapted ones.

    score = 0.5 * cos(centroid) + 0.5 * mean of the top-5 cosines to individual
    embeddings. The centroid is robust; the top-k part rewards matching any one
    of your recorded conditions (fan on, leaning back, soft voice...).
    """

    def __init__(self, model_name: str, enroll: np.ndarray, adapted: np.ndarray | None = None):
        self.model_name = model_name
        self.enroll = normalize(np.asarray(enroll, dtype=np.float32))
        dim = self.enroll.shape[1]
        self.adapted = np.zeros((0, dim), np.float32) if adapted is None else normalize(adapted)

    @property
    def bank(self) -> np.ndarray:
        return np.vstack([self.enroll, self.adapted])

    @property
    def centroid(self) -> np.ndarray:
        return normalize(self.enroll.mean(axis=0))

    def score(self, emb: np.ndarray) -> float:
        emb = normalize(emb)
        c = float(self.centroid @ emb)
        sims = np.sort(self.bank @ emb)[::-1]
        return 0.5 * c + 0.5 * float(sims[:5].mean())

    def adapt(self, emb: np.ndarray, max_size: int) -> None:
        self.adapted = np.vstack([self.adapted, normalize(emb)[None]])[-max_size:]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, model_name=self.model_name, enroll=self.enroll, adapted=self.adapted)

    @classmethod
    def load(cls, path: str | Path) -> "OwnerProfile":
        d = np.load(path)
        return cls(str(d["model_name"]), d["enroll"], d["adapted"])
