"""Neural Encoding Component (realises report Section 6.5.3).

Supplies the topic-coherence signal the hybrid merge uses to decide whether two
adjacent micro-chunks belong together: encode each with LaBSE, L2-normalise, and
return the dot product — a cosine similarity in ``[-1, 1]``.

Two details follow the report rather than the prototype:

- **L2-normalise then dot product**, not ``sklearn.cosine_similarity`` on raw
  vectors. Same number, but normalising once per chunk and caching the result
  means the similarity itself is a single multiply.
- **Stateless with respect to the model**, so "the multilingual encoder can be
  replaced by a Sinhala-specialised variant without any change to the calling
  code" (§6.5.3). ``model_name`` is a constructor argument; a fine-tuned LaBSE is
  a one-line swap.

The prototype re-encoded the growing accumulator on every merge step, which is
O(n) encodes for an n-micro-chunk document. Embeddings are cached by text here,
so each distinct span is encoded once.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

from akshara_kit.eye.errors import AdapterUnavailableError

__all__ = ["DEFAULT_MODEL", "CoherenceScorer", "LabseScorer"]

#: The stock multilingual encoder. Sinhala is among LaBSE's 109 languages, so
#: this works without any fine-tuning; replace with a fine-tuned checkpoint by
#: passing ``model_name``, or via ``AKSHARA_LABSE_MODEL`` — see
#: :class:`LabseScorer`.
DEFAULT_MODEL = "setu4993/LaBSE"


@runtime_checkable
class CoherenceScorer(Protocol):
    """Scores how strongly two spans discuss the same topic.

    The coordinator depends on this, not on LaBSE, which is what lets the whole
    merge stage be tested with a stub — no torch, no model download, no network.
    """

    def score(self, left: str, right: str) -> float:
        """Similarity in ``[-1, 1]``; higher means more topically alike."""
        ...


class LabseScorer:
    """:class:`CoherenceScorer` backed by a sentence-transformers model.

    Model resolution, in order: the ``model_name`` argument, then the
    ``AKSHARA_LABSE_MODEL`` environment variable, then :data:`DEFAULT_MODEL`.
    This mirrors the ``AKSHARA_*`` override already used for Tesseract and
    SWI-Prolog in :mod:`akshara_kit.eye.capabilities` — a fine-tuned checkpoint
    can be selected without touching code, and the library still works with
    nothing but the public hub model on a machine that has no such checkpoint.
    """

    def __init__(self, model_name: str | None = None, *, cache_size: int = 4096):
        self.model_name = model_name or os.environ.get("AKSHARA_LABSE_MODEL") or DEFAULT_MODEL
        self._cache_size = cache_size
        self._model = None
        self._cache: dict[str, object] = {}

    def _load(self):
        """Load the encoder once, on first use rather than at construction.

        Deferred so that importing the Brain does not pull in torch, and so a
        caller using a stub scorer never pays for a model they will not call.
        """
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise AdapterUnavailableError(
                "sentence-transformers is not installed; install the 'brain' extra"
            ) from exc

        self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, text: str):
        """L2-normalised embedding for one span, cached by text."""
        if text in self._cache:
            return self._cache[text]

        import numpy as np

        vector = self._load().encode(text, convert_to_numpy=True)
        norm = float(np.linalg.norm(vector))
        # A zero vector has no direction; returning it unnormalised keeps the
        # dot product at 0.0 rather than producing a NaN that would silently
        # poison every comparison downstream.
        normalised = vector / norm if norm else vector

        if len(self._cache) >= self._cache_size:
            self._cache.clear()
        self._cache[text] = normalised
        return normalised

    def score(self, left: str, right: str) -> float:
        """Cosine similarity, as the dot product of two unit vectors."""
        if not left.strip() or not right.strip():
            return 0.0

        import numpy as np

        return float(np.dot(self.embed(left), self.embed(right)))
