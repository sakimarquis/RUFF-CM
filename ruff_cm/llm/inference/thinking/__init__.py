"""HF and OpenAI-compatible thinking-runtime helpers."""

from .api import two_stage_api_call
from .codec import HfThinkingCodec
from .flow import two_step_hf_flow
from .processor import RowState, ThinkingBudgetProcessor, _AllCaptured, _CapturePostThinkLogits, recover_uncaptured_logits
from .protocol import ThinkingProtocol, resolve_thinking_protocol

__all__ = [
    "HfThinkingCodec",
    "RowState",
    "ThinkingBudgetProcessor",
    "ThinkingProtocol",
    "_AllCaptured",
    "_CapturePostThinkLogits",
    "recover_uncaptured_logits",
    "resolve_thinking_protocol",
    "two_stage_api_call",
    "two_step_hf_flow",
]
