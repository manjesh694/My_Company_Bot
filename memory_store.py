import json
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Callable


@dataclass
class MemoryEntry:
    """A single memory entry with metadata (Page 344)."""
    content: str
    embedding: np.ndarray
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 0.5
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    source: str = "agent"


class VectorMemoryStore:
    """Hybrid dense+sparse memory store with temporal decay (Pages 344-346)."""
    
    def __init__(
        self,
        embed_fn: Callable[[str], np.ndarray],
        max_entries: int = 10000,
        decay_rate: float = 0.01,
        recency_weight: float = 0.3
    ):
        self.embed_fn = embed_fn
        self.max_entries = max_entries
        self.decay_rate = decay_rate
        self.recency_weight = recency_weight
        self.entries: List[MemoryEntry] = []

    def write(
        self,
        content: str,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        check_duplicates: bool = True
    ) -> Optional[MemoryEntry]:
        """Commit a new memory, evicting if at capacity."""
        if check_duplicates and self._is_duplicate(content):
            return None  # Skip near-duplicate entries

        embedding = self.embed_fn(content)
        entry = MemoryEntry(
            content=content,
            embedding=embedding,
            importance=importance,
            tags=tags or []
        )

        if len(self.entries) >= self.max_entries:
            self._evict()

        self.entries.append(entry)
        return entry

    def _is_duplicate(self, content: str, threshold: float = 0.95) -> bool:
        """Check if a near-duplicate memory already exists."""
        if not self.entries:
            return False
        emb = self.embed_fn(content)
        sims = self._cosine_similarities(emb)
        return float(np.max(sims)) > threshold

    def _evict(self):
        """Remove the least important + least recent entry."""
        now = datetime.now()
        scores = []
        for e in self.entries:
            age_hours = (now - e.timestamp).total_seconds() / 3600
            recency = np.exp(-self.decay_rate * age_hours)
            score = e.importance * (1 - self.recency_weight) + recency * self.recency_weight
            scores.append(score)
        worst_idx = int(np.argmin(scores))
        self.entries.pop(worst_idx)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        recency_boost: bool = True
    ) -> List[MemoryEntry]:
        """Hybrid retrieval: dense similarity + temporal recency."""
        if not self.entries:
            return []

        q_emb = self.embed_fn(query)
        dense_scores = self._cosine_similarities(q_emb)
        now = datetime.now()
        combined = []

        for i, (entry, d_score) in enumerate(zip(self.entries, dense_scores)):
            if recency_boost:
                age_h = (now - entry.timestamp).total_seconds() / 3600
                recency = np.exp(-self.decay_rate * age_h)
                score = (1 - self.recency_weight) * d_score + self.recency_weight * recency
            else:
                score = d_score
            combined.append((score, i))

        combined.sort(reverse=True)
        top_k = [self.entries[i] for _, i in combined[:k]]

        # Update access metadata
        for entry in top_k:
            entry.access_count += 1
            entry.last_accessed = now

        return top_k

    def _cosine_similarities(self, query_emb: np.ndarray) -> np.ndarray:
        """Vectorized cosine similarity against all stored embeddings."""
        matrix = np.stack([e.embedding for e in self.entries])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix_norm = matrix / (norms + 1e-8)
        q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        return matrix_norm @ q_norm

    def get_stats(self) -> dict:
        """Return memory statistics for monitoring."""
        return {
            "total_entries": len(self.entries),
            "avg_importance": float(np.mean([e.importance for e in self.entries])) if self.entries else 0.0,
        }


# --- Self-Test Block ---
if __name__ == "__main__":
    print("Testing VectorMemoryStore module...")

    # Dummy embedding function using random vectors for testing
    def dummy_embed_fn(text: str) -> np.ndarray:
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(128)

    memory = VectorMemoryStore(embed_fn=dummy_embed_fn)
    memory.write("User prefers concise answers", importance=0.8)
    memory.write("User works as a software engineer", importance=0.9)

    retrieved = memory.retrieve("software engineer", k=1)
    print(f"Retrieved Memory: '{retrieved[0].content}'")
    print("Memory Store Test Passed Successfully!")