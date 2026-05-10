"""Forward execution, KV-cache plumbing, thinking runtime, and batch request scaffolding."""

from .batch import JobManifest, RequestRecord, collect_ordered_results
from .execution import (
    forward_hidden_only,
    forward_query_logits,
    forward_selected_logits,
    resolve_base_forward_model,
    resolve_decoder_layers,
)
from .generate import (
    ParseFailure,
    ParseRunReport,
    generate_and_parse,
    print_parse_failure_summary,
    save_parse_failure_report,
)
from .kvcache import (
    HybridCacheAdapter,
    clone_kv,
    concat_kv,
    forward_with_kv_delta,
    is_hybrid_supported,
    kv_seq_len,
    reposition_kv,
    tail_kv,
    to_dynamic_cache,
    to_legacy_kv,
    truncate_kv,
)
from .latent import LatentThoughtResult, apply_alignment, compute_alignment_matrix, generate_latent_thoughts
from .retry import is_transient_api_error, with_api_retry, with_oom_halving, with_parse_retry
from .runtime import InferenceResult, generate
from .specs import BudgetSpec, FinishReason, SamplingConfig, ScoringSpec
from . import thinking

__all__ = [
    "BudgetSpec",
    "FinishReason",
    "HybridCacheAdapter",
    "InferenceResult",
    "JobManifest",
    "LatentThoughtResult",
    "ParseFailure",
    "ParseRunReport",
    "RequestRecord",
    "SamplingConfig",
    "ScoringSpec",
    "apply_alignment",
    "clone_kv",
    "compute_alignment_matrix",
    "collect_ordered_results",
    "concat_kv",
    "forward_hidden_only",
    "forward_query_logits",
    "forward_selected_logits",
    "forward_with_kv_delta",
    "generate",
    "generate_latent_thoughts",
    "generate_and_parse",
    "is_hybrid_supported",
    "is_transient_api_error",
    "print_parse_failure_summary",
    "kv_seq_len",
    "reposition_kv",
    "resolve_base_forward_model",
    "resolve_decoder_layers",
    "save_parse_failure_report",
    "thinking",
    "tail_kv",
    "to_dynamic_cache",
    "to_legacy_kv",
    "truncate_kv",
    "with_api_retry",
    "with_oom_halving",
    "with_parse_retry",
]
