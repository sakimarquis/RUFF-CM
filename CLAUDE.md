# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Start Here

- Read `./README.md` when you need to understand repo structure, public APIs, or workflow.
- Treat this package as shared infrastructure for downstream research repos; preserve behavior unless the user explicitly asks for a behavior change.
- Keep responses concise and technical.

## Repo Map

- `ruff_cm.llm`: LLM backend protocols, API/HF implementations, choice scoring, hidden capture, and thinking config.
- `ruff_cm.experimenter`: config-grid helpers and explicit experiment-cell identity.
- `ruff_cm.store`: artifact fingerprints, paths, sidecar metadata, and stale-artifact detection.
- `tests/parity`: downstream behavior fixtures; update them only when the corresponding downstream contract intentionally changes.

## Public Surface

- Keep exported names in `ruff_cm/*/__init__.py` aligned with the README and tests when changing public APIs.
- `create_backend` currently instantiates `api` and `hf` aliases. `google_cloud` alias metadata is consumed by `resolve_thinking`, not by backend construction.
- `ChoiceSet` is intentionally scoped to single-token candidates. Multi-token choice scoring should fail clearly until it is designed as a real feature.
- `HiddenCapture` returns tensors shaped as `batch x positions x hidden_dim` for selected positions; preserve that contract.
- Prefer adding focused parity tests when extracting behavior from downstream repos.

## Coding Principles

- Make **minimal changes** that preserve intent and improve clarity; prefer surfacing real bugs over masking them.
- **Abstractions for reuse, not labeling**: extract shared logic to capture symmetry and reduce duplication — never copy-paste. Do not wrap for "conceptual clarity" alone: a forwarding wrapper tells you *what* but hides *how*, adding indirection without reducing complexity. Real compression produces clarity as a side effect — that is why good abstractions happen to be readable. Litmus test: if you delete it, does duplication appear or symmetry break? It earns its place. Nothing changes? It's empty — inline it. When refactoring or extracting abstractions, preserve all behavior including performance. When uncertain, ask.
- **Let it fail**: use an **optimistic, let-it-crash research style**.
  Do not add defensive checks, try/except blocks, or fallbacks unless explicitly required.
  Assume invariants hold; let violations crash early and visibly.
- **Formatting**: preserve **semantic units**.
  Keep single conceptual operations on one line when readable; avoid one-arg-per-line vertical formatting unless it adds information (e.g., config blocks).
- Prefer readable, maintainable code over clever tricks or premature optimization.
- Separate side effects from pure logic when it improves clarity; avoid over-modularizing trivial code.
- Make data formats explicit when helpful (e.g., tensor shapes), but keep it light.
- Functions longer than 10 lines or conceptually complex must include short comments explaining **intent or non-obvious reasoning**.
- Fix root causes (not band-aids). If unsure: read more code, then ask.
- **No historical comments**: comments describe the present, not what code used to be.
- **Naming**: names should be immediately clear, informative enough to convey purpose, no longer than necessary, and consistent across the codebase.
- Unrecognized changes: assume other agent; keep going; focus your changes. If it causes issues, stop + ask user.
- No abstractions for single-use code.
- No error handling for impossible scenarios.
- Don't hide confusion.
- Surface tradeoffs.
- State assumptions, flag alternatives, ask when unclear.

## Performance

- Preserve existing lazy-loading behavior for heavyweight LLM dependencies.
- Optimize only when it improves clarity or runtime without changing downstream contracts.
