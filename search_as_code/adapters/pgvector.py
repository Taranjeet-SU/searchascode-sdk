"""pgvector adapter. ``pip install 'search-as-code[pgvector]'``.

Postgres + the pgvector extension.  Metadata lives in a JSONB column; the
portable filter dialect is translated to parameterized SQL.  Cosine *distance*
(``<=>``) is converted to a larger-is-better similarity.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Sequence

from ..filters import normalize
from ..types import Capabilities, Document, Hit, ResultSet
from .base import VectorStore

_SQL_OP = {"$eq": "=", "$ne": "<>", "$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}


class PgVectorStore(VectorStore):
    backend = "pgvector"

    def __init__(self, dsn: str, table: str = "sac_docs", dim: int = 256, **_: Any):
        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError("pip install 'search-as-code[pgvector]'") from e
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self._conn)
        self.table = table
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {table} "
            f"(id TEXT PRIMARY KEY, text TEXT, metadata JSONB, embedding vector({dim}))"
        )

    def capabilities(self) -> Capabilities:
        # Postgres also has full-text search; wire query_keyword() to tsvector to
        # flip `keyword`/`hybrid` on. Left dense-only for the reference adapter.
        return Capabilities(dense=True, keyword=False, hybrid=False, metadata_filter=True)

    def upsert(self, docs: Sequence[Document]) -> None:
        rows = [
            (d.id, d.text, json.dumps(d.metadata or {}), d.vector)
            for d in docs
            if d.vector is not None
        ]
        if not rows:
            return
        with self._conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {self.table} (id, text, metadata, embedding) VALUES (%s,%s,%s,%s) "
                f"ON CONFLICT (id) DO UPDATE SET text=EXCLUDED.text, "
                f"metadata=EXCLUDED.metadata, embedding=EXCLUDED.embedding",
                rows,
            )

    def _where(self, flt: Optional[dict]) -> tuple[str, list]:
        if not flt:
            return "", []
        clauses, params = [], []
        for field_name, cond in normalize(flt).items():
            if field_name.startswith("$"):
                continue
            for op, val in cond.items():
                if op in _SQL_OP:
                    clauses.append(f"metadata->>%s {_SQL_OP[op]} %s")
                    params.extend([field_name, str(val)])
                elif op == "$in":
                    clauses.append(f"metadata->>%s = ANY(%s)")
                    params.extend([field_name, [str(v) for v in val]])
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    def query_vector(self, vector, top_k=10, flt=None) -> ResultSet:
        import numpy as np

        where, params = self._where(flt)
        sql = (
            f"SELECT id, text, metadata, 1 - (embedding <=> %s) AS score "
            f"FROM {self.table}{where} ORDER BY embedding <=> %s LIMIT %s"
        )
        vec = np.asarray(vector, dtype=np.float32)
        rows = self._conn.execute(sql, [vec, *params, vec, top_k]).fetchall()
        return ResultSet(
            Hit(
                id=r[0],
                score=float(r[3]),
                document=Document(id=r[0], text=r[1], metadata=r[2] or {}),
                store=self.backend,
            )
            for r in rows
        )

    def get(self, ids: Sequence[str]) -> list[Document]:
        rows = self._conn.execute(
            f"SELECT id, text, metadata FROM {self.table} WHERE id = ANY(%s)", [list(ids)]
        ).fetchall()
        return [Document(id=r[0], text=r[1], metadata=r[2] or {}) for r in rows]

    def delete(self, ids: Sequence[str]) -> None:
        self._conn.execute(f"DELETE FROM {self.table} WHERE id = ANY(%s)", [list(ids)])

    def count(self) -> int:
        return self._conn.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0]

    def close(self) -> None:
        self._conn.close()
