"""Reciprocal Rank Fusion for merging per-domain result lists."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from agent_service.kb.base import SearchHit


def reciprocal_rank_fusion(
    results: Mapping[str, Sequence[SearchHit]], *, k: int = 60, top_k: int = 5
) -> list[SearchHit]:
    """Merge ranked lists from several domains into one ranking.

    RRF scores a document as ``sum(1 / (k + rank))`` across the lists it appears
    in, which is robust when the underlying scores are not comparable (different
    domains, different collections).
    """
    fused: dict[str, tuple[float, SearchHit]] = {}
    for domain, hits in results.items():
        for rank, hit in enumerate(hits, start=1):
            key = f"{domain}:{hit.id}"
            contribution = 1.0 / (k + rank)
            if key in fused:
                previous_score, previous_hit = fused[key]
                fused[key] = (previous_score + contribution, previous_hit)
            else:
                fused[key] = (contribution, hit)

    ranked = sorted(fused.values(), key=lambda item: item[0], reverse=True)
    merged: list[SearchHit] = []
    for score, hit in ranked[: max(0, top_k)]:
        merged.append(
            hit.model_copy(
                update={
                    "metadata": {**hit.metadata, "raw_score": hit.score},
                    "score": round(score, 6),
                }
            )
        )
    return merged
