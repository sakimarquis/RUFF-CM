## Start Here
- Read `./README.md` when you need to understand repo structure + workflow.
- Consult local memory first if available.
- Keep responses concise and technical.

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

## Performance
- If you notice meaningful loop/condition/data-structure optimizations or clear performance/readability improvements, tell the user and include them in the plan.
- Optimize when it improves clarity or performance.
