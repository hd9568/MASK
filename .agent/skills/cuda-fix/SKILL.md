---
name: "cuda-fix"
description: "Repairs compile or runtime failures in generated CUDA kernels. Invoke when a generated kernel fails to compile, crashes, or needs a minimal targeted patch."
---

# CUDA Fix

## Mission

Use this skill only after a generated kernel has already failed and a minimal repair is needed.

## Scope

- compile errors
- missing identifiers
- template misuse
- narrow runtime-failure fixes that can be localized to generated code

## Rules

- Patch only the specific failing lines.
- Avoid refactoring unrelated code.
- Preserve the intended tensor layout and pipeline behavior unless the failure proves they are invalid.
- Prefer deterministic, minimal text replacements that are easy to audit.

## Inputs You Need

- failing kernel source
- compiler or runtime error log
- current iteration context if available

## Output Style

Return a small JSON patch plan with:

- `action`
- `patches`
- `rationale`

Abort if the failure cannot be fixed safely with a targeted patch.
