# `solvers/`

Per-category Python solvers for Phase 2. Each solver takes a puzzle's `prompt`
and ground-truth-supporting examples, brute-force searches a hypothesis space
of candidate rules, and returns the discovered rule (or `None` if it fails).

The discovered rule is then fed to a teacher LLM (DeepSeek V3.2) which
generates a clean chain-of-thought trace using the rule as scaffolding. The
trace is filtered through `eval.grader.verify` and added to the SFT mix.

## Planned solvers

- `transformation.py` — symbolic rule discovery: index permutations, arithmetic
  ops, modular maps, base-N transforms, character-class substitutions, and
  two-stage compositions. **Highest leverage** -- 399 transformation puzzles
  in validation are unsolved by all public teachers.
- `bit_manipulation.py` — enumerate Boolean-function families up to 4 inputs,
  depth 3.
- `cipher.py` — classic ciphers (Caesar, ROT-N, Atbash, Vigenere, monoalphabetic
  substitution, columnar / rail-fence, Base-N) with small-keyspace brute force.
- `numeral.py`, `unit_conversion.py`, `gravity.py` — backfill solvers for
  cases the existing andy279 traces missed.

## Status

To be implemented in Phase 2 (days 7-22). This directory is intentionally
empty during Phase 0/1.
