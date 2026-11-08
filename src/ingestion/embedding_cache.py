"""On-disk cache of document embeddings, keyed by (model, sha256(text)).

Embedding is the slow step (~9 docs/s on a 4-core laptop CPU), so caching makes
re-ingestion of unchanged products near-free and lets an interrupted run resume.
SQLite: stdlib, single file, safe to stop at any point (each batch is a transaction).
"""
import hashlib
import sqlite3
from pathlib import Path

import numpy as np

DEFAULT_PATH = Path(__file__).resolve().parents[2] / ".cache" / "embeddings.sqlite"


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    def __init__(self, path: Path | str = DEFAULT_PATH, model_tag: str = "") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.model_tag = model_tag
        self.conn = sqlite3.connect(self.path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS emb ("
            " model TEXT NOT NULL, key TEXT NOT NULL,"
            " dense BLOB NOT NULL, sp_idx BLOB NOT NULL, sp_val BLOB NOT NULL,"
            " PRIMARY KEY (model, key))"
        )
        self.conn.commit()

    def get_many(self, texts: list[str]) -> dict[str, tuple[list[float], dict[str, list]]]:
        keys = [text_key(t) for t in texts]
        found: dict[str, tuple[list[float], dict[str, list]]] = {}
        for i in range(0, len(keys), 500):  # stay under SQLite's parameter limit
            chunk = keys[i:i + 500]
            rows = self.conn.execute(
                f"SELECT key, dense, sp_idx, sp_val FROM emb WHERE model = ? AND key IN ({','.join('?' * len(chunk))})",
                [self.model_tag, *chunk],
            ).fetchall()
            for key, dense, sp_idx, sp_val in rows:
                found[key] = (
                    np.frombuffer(dense, dtype=np.float32).tolist(),
                    {"indices": np.frombuffer(sp_idx, dtype=np.int64).tolist(),
                     "values": np.frombuffer(sp_val, dtype=np.float32).tolist()},
                )
        return found

    def put_many(self, texts: list[str], dense: list[list[float]], sparse: list[dict]) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO emb VALUES (?, ?, ?, ?, ?)",
            [
                (self.model_tag, text_key(t),
                 np.asarray(d, dtype=np.float32).tobytes(),
                 np.asarray(s["indices"], dtype=np.int64).tobytes(),
                 np.asarray(s["values"], dtype=np.float32).tobytes())
                for t, d, s in zip(texts, dense, sparse, strict=True)
            ],
        )
        self.conn.commit()

    def __len__(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM emb WHERE model = ?", [self.model_tag]).fetchone()[0]

    def close(self) -> None:
        self.conn.close()
