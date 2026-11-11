"""Frozen, provenance-preserving sources for obligation-bound COMPLETE.

The catalog is deliberately small and boring.  It does not search by the QA
question and it never sees a gold answer.  A lookup is keyed only by the entity
already bound by the physical query plan.  Returned passages still go through
the normal tuple extraction, obligation admission, and source verification
pipeline.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


CATALOG_VERSION = "frozen_obligation_source_catalog_v1"
SYNTHETIC_SOURCE_ID_BASE = 1_800_000_000
SYNTHETIC_SOURCE_ID_SPAN = 300_000_000


def _title_key(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(re.findall(r"[^\W_]+", text, flags=re.UNICODE))


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _stable_source_id(title: str, revision_id: str, content_hash: str) -> int:
    material = "\x1e".join((title, revision_id, content_hash))
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return SYNTHETIC_SOURCE_ID_BASE + int(digest[:15], 16) % SYNTHETIC_SOURCE_ID_SPAN


class FrozenObligationSourceCatalog:
    """Read-only exact-title catalog used after a local obligation miss."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self._by_title: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._source_ids: Dict[int, str] = {}
        if not self.path.is_file():
            raise FileNotFoundError(f"Frozen source catalog not found: {self.path}")
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            raw = json.loads(line)
            title = str(raw.get("title") or "").strip()
            content = str(raw.get("content") or "").strip()
            if not title or not content:
                raise ValueError(
                    f"Catalog row {line_number} requires non-empty title/content"
                )
            expected_hash = str(raw.get("content_sha256") or "")
            actual_hash = _content_sha256(content)
            if expected_hash and expected_hash != actual_hash:
                raise ValueError(
                    f"Catalog row {line_number} content hash mismatch"
                )
            revision_id = str(raw.get("revision_id") or "").strip()
            source_id = int(
                raw.get("source_id")
                or _stable_source_id(title, revision_id, actual_hash)
            )
            prior_hash = self._source_ids.get(source_id)
            if prior_hash and prior_hash != actual_hash:
                raise ValueError(f"Synthetic source-id collision: {source_id}")
            self._source_ids[source_id] = actual_hash
            row = {
                "doc_id": source_id,
                "source_id": source_id,
                "title": title,
                "content": content,
                "source_url": str(raw.get("source_url") or ""),
                "revision_id": revision_id,
                "content_sha256": actual_hash,
                "catalog_version": str(
                    raw.get("catalog_version") or CATALOG_VERSION
                ),
                "source_scope": str(raw.get("source_scope") or ""),
                "predicate_families": list(raw.get("predicate_families") or []),
                "required_predicate_subtypes": list(
                    raw.get("required_predicate_subtypes") or []
                ),
                "relation_tokens": list(raw.get("relation_tokens") or []),
                "relation_hit_count": int(raw.get("relation_hit_count") or 0),
                "selected_sentence_count": int(
                    raw.get("selected_sentence_count") or 0
                ),
                "completion_source": "frozen_external_title_catalog",
            }
            keys = {_title_key(title)}
            keys.update(
                _title_key(alias)
                for alias in (raw.get("aliases") or [])
                if _title_key(alias)
            )
            for key in sorted(keys):
                self._by_title[key].append(dict(row))

    def lookup(
        self,
        bound_surface: str,
        operator: Mapping[str, Any],
        limit: int,
    ) -> Sequence[Mapping[str, Any]]:
        """Return exact entity-title pages; the relation is not pre-answered."""
        key = _title_key(bound_surface)
        if not key or limit <= 0:
            return []
        operator_families = {
            str(value).casefold()
            for value in operator.get("predicate_families") or []
        }
        operator_subtypes = {
            str(value).casefold()
            for value in operator.get("required_predicate_subtypes") or []
        }
        ranked = []
        for raw in self._by_title.get(key) or []:
            row_families = {
                str(value).casefold()
                for value in raw.get("predicate_families") or []
            }
            row_subtypes = {
                str(value).casefold()
                for value in raw.get("required_predicate_subtypes") or []
            }
            if row_subtypes and operator_subtypes and not (
                row_subtypes & operator_subtypes
            ):
                continue
            if row_families and operator_families and not (
                row_families & operator_families
            ):
                continue
            score = (
                2 * len(row_subtypes & operator_subtypes)
                + len(row_families & operator_families)
                + int(not row_subtypes and not row_families)
            )
            ranked.append((score, str(raw.get("content_sha256") or ""), raw))
        rows = []
        for rank, (match_score, _, raw) in enumerate(
            sorted(ranked, key=lambda value: (-value[0], value[1]))
        ):
            row = dict(raw)
            row["rank"] = rank
            row["score"] = float(match_score)
            row["bound_title_key"] = key
            rows.append(row)
            if len(rows) >= limit:
                break
        return rows


__all__ = [
    "CATALOG_VERSION",
    "FrozenObligationSourceCatalog",
]
