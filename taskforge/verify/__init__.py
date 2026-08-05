"""Domain-agnostic three-stage verifier: schema (V1), oracle (V2), replay (V3)."""

from taskforge.verify.pipeline import (
    BUCKET_EDGES,
    Certificate,
    Difficulty,
    VerificationOutcome,
    VerifiedTask,
    accepted_task,
    bucket_for,
    difficulty_of,
    load_specs,
    revalidate,
    sku_scatter,
    verify,
)
from taskforge.verify.v1_schema import StageResult, check_v1
from taskforge.verify.v2_oracle import (
    CostToGo,
    OracleResult,
    OracleStatus,
    cost_to_go,
    plan_signature,
    positional_cost_to_go,
    solve,
)
from taskforge.verify.v3_replay import ReplayTrace, check_v3, replay

__all__ = [
    "BUCKET_EDGES",
    "Certificate",
    "CostToGo",
    "Difficulty",
    "OracleResult",
    "OracleStatus",
    "ReplayTrace",
    "StageResult",
    "VerificationOutcome",
    "VerifiedTask",
    "accepted_task",
    "bucket_for",
    "check_v1",
    "check_v3",
    "cost_to_go",
    "difficulty_of",
    "load_specs",
    "plan_signature",
    "positional_cost_to_go",
    "replay",
    "revalidate",
    "sku_scatter",
    "solve",
    "verify",
]
