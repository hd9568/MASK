"""Auditable executable siblings for a saved research-format proposal.

The format-design agent owns the original DSL.  This module never rewrites
that decision in place.  Instead, when a proposal reaches an implemented
SM80 two-level column-condensed backend, it materializes semantically matched
atomic siblings so the feedback loop can measure a real MCO-versus-dense
kernel trade-off.  This is necessary because storage estimates alone do not
model MCO restore, B gathers, or Tensor Core work inside the timed kernel.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from sparsene.format.format import Format
from sparsene.formats.dsl import build_format_from_spec, validate_format_spec
from sparsene.op_gen.backend_registry import BackendPlan, select_backend


@dataclass(frozen=True)
class ExecutableFormatCandidate:
    """One independently generated executable candidate.

    ``source`` deliberately distinguishes an Agent proposal from a
    deterministic measurement sibling.  Callers must report this field and
    must not relabel a sibling as an Agent-selected format.
    """

    candidate_id: str
    source: str
    atomic_type: str
    spec: dict[str, Any]
    format_obj: Format
    backend_plan: BackendPlan

    @property
    def spec_sha256(self) -> str:
        encoded = json.dumps(
            self.spec, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def provenance(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source": self.source,
            "atomic_type": self.atomic_type,
            "format_spec_sha256": self.spec_sha256,
            "backend": self.backend_plan.backend,
            "codegen_format_name": self.backend_plan.codegen_format_name,
        }


def _candidate(
    *, candidate_id: str, source: str, spec: Mapping[str, Any]
) -> ExecutableFormatCandidate:
    materialized = deepcopy(dict(spec))
    validation = validate_format_spec(materialized)
    if validation["status"] != "valid":
        raise ValueError(
            f"{candidate_id}: invalid format DSL: {'; '.join(validation['errors'])}"
        )
    plan = select_backend(materialized, target="sm80")
    return ExecutableFormatCandidate(
        candidate_id=candidate_id,
        source=source,
        atomic_type=str(materialized["atomic"]["type"]),
        spec=materialized,
        format_obj=build_format_from_spec(materialized),
        backend_plan=plan,
    )


def executable_atomic_portfolio(
    spec: Mapping[str, Any],
) -> list[ExecutableFormatCandidate]:
    """Return the original proposal plus valid MCO/dense measurement siblings.

    Only the implemented two-level column-condensed hierarchy gets a sibling.
    A one-level BIT-BSR proposal intentionally remains alone: changing its
    leaf to dense would be a different unimplemented host/backend contract.
    """

    original = _candidate(candidate_id="agent", source="agent_proposal", spec=spec)
    candidates = [original]
    if (
        original.backend_plan.packing_mode != "column_condensed"
        or original.atomic_type not in {"mco", "dense"}
    ):
        return candidates

    sibling_atomic = "dense" if original.atomic_type == "mco" else "mco"
    sibling_spec = deepcopy(original.spec)
    sibling_spec["atomic"]["type"] = sibling_atomic
    sibling_spec["atomic"]["rationale"] = (
        "Deterministic executable atomic sibling for kernel-only portfolio "
        "measurement; the original Agent proposal is preserved unchanged."
    )
    sibling_spec["atomic"]["custom_spec"] = None
    candidates.append(
        _candidate(
            candidate_id=f"atomic_{sibling_atomic}_sibling",
            source="deterministic_atomic_sibling",
            spec=sibling_spec,
        )
    )
    return candidates


def executable_tcf_portfolio(
    spec: Mapping[str, Any],
) -> list[ExecutableFormatCandidate]:
    """Add a compact hardware-aligned dense tile portfolio to atomic siblings.

    Logical M=8 is physically padded to 16 by the SM80 backend, while a
    dense TCF tile has no MCO restore path to compensate for that cost.  These
    three candidates make that architectural trade-off measurable without
    changing the saved Agent proposal.  They are deliberately a small
    validation portfolio rather than an unbounded tile autotuner.
    """

    candidates = executable_atomic_portfolio(spec)
    original = candidates[0]
    if original.backend_plan.packing_mode not in {
        "column_condensed",
        "column_condensed_dense",
    }:
        return candidates

    existing = {
        (candidate.atomic_type, int(candidate.spec["params"]["BLK_M"]),
         int(candidate.spec["params"]["BLK_K"]))
        for candidate in candidates
    }
    for tile_m, tile_k in ((16, 8), (16, 16), (32, 8)):
        key = ("dense", tile_m, tile_k)
        if key in existing:
            continue
        sibling_spec = deepcopy(original.spec)
        sibling_spec["params"]["BLK_M"] = tile_m
        sibling_spec["params"]["BLK_K"] = tile_k
        sibling_spec["atomic"]["type"] = "dense"
        sibling_spec["atomic"]["rationale"] = (
            "Deterministic hardware-aligned dense TCF sibling for kernel-only "
            "portfolio measurement; the original Agent proposal is preserved unchanged."
        )
        sibling_spec["atomic"]["custom_spec"] = None
        candidates.append(
            _candidate(
                candidate_id=f"dense_{tile_m}x{tile_k}_sibling",
                source="deterministic_tile_atomic_sibling",
                spec=sibling_spec,
            )
        )
    return candidates


def select_portfolio_candidate(
    spec: Mapping[str, Any], candidate_id: str
) -> ExecutableFormatCandidate:
    """Find one candidate by stable ID or fail without falling back."""

    for candidate in executable_tcf_portfolio(spec):
        if candidate.candidate_id == candidate_id:
            return candidate
    available = ", ".join(item.candidate_id for item in executable_tcf_portfolio(spec))
    raise ValueError(
        f"No executable portfolio candidate {candidate_id!r}; available: {available}"
    )
