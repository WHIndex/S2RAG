from typing import Any, Dict


def entity_bound_join_key_match(active_keys: Any, candidate_keys: Any) -> Dict[str, Any]:
    """Match role-aware fact endpoints without generic-token containment leaks."""
    if isinstance(active_keys, list):
        active_keys = {"any": active_keys}
    if isinstance(candidate_keys, list):
        candidate_keys = {"any": candidate_keys}
    active_keys = active_keys or {}
    candidate_keys = candidate_keys or {}

    best = {
        "overlap": 0.0,
        "containment": 0.0,
        "exact": False,
        "bound": False,
        "active_role": "",
        "candidate_role": "",
        "active_key_tokens": [],
        "candidate_key_tokens": [],
        "matched_tokens": [],
    }
    for active_role, active_values in active_keys.items():
        for candidate_role, candidate_values in candidate_keys.items():
            for active_raw in active_values or []:
                active_key = set(active_raw or [])
                for candidate_raw in candidate_values or []:
                    candidate_key = set(candidate_raw or [])
                    if not active_key or not candidate_key:
                        continue
                    intersection = active_key & candidate_key
                    containment = len(intersection) / max(
                        1,
                        min(len(active_key), len(candidate_key)),
                    )
                    overlap = len(intersection) / max(1, len(active_key | candidate_key))
                    exact = active_key == candidate_key
                    bound = bool(exact or (len(intersection) >= 2 and overlap >= 0.66))
                    candidate_result = {
                        "overlap": float(overlap),
                        "containment": float(containment),
                        "exact": bool(exact),
                        "bound": bool(bound),
                        "active_role": str(active_role),
                        "candidate_role": str(candidate_role),
                        "active_key_tokens": sorted(active_key),
                        "candidate_key_tokens": sorted(candidate_key),
                        "matched_tokens": sorted(intersection),
                    }
                    if (
                        int(candidate_result["bound"]),
                        candidate_result["overlap"],
                        len(candidate_result["matched_tokens"]),
                    ) > (
                        int(best["bound"]),
                        best["overlap"],
                        len(best["matched_tokens"]),
                    ):
                        best = candidate_result
    return best
