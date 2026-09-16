# Leakage Audit Template

## Scope

- Dataset:
- Source identifier:
- Processed artifact:
- Config file:
- Run timestamp:

## Checks Performed

- Timestamp column exists and parses successfully.
- Timestamp values are non-null.
- Timestamp values are unique.
- Input rows are already in chronological order before splitting.
- Train/validation/test fractions are valid and deterministic.
- Train/validation gap is measured in timestamp time, not row counts.
- Validation/test gap is measured in timestamp time, not row counts.
- Raw dataset provenance records source, filename, size, checksum, conversion time, row count, and schema summary.

## Checks Passed

- To be completed by the audit run.

## Checks Not Yet Possible

- Window boundary leakage, until time-window generation is implemented.
- Event/failure leakage, until failure intervals are represented in code.
- Threshold leakage, until validation-only threshold selection is implemented.
- Preprocessing leakage beyond split boundaries, until training orchestration is implemented.

## Assumptions

- The timestamp column configured for the run is the correct temporal ordering field.
- The raw dataset is preserved separately and is not modified by conversion.
- Dataset-specific required sensor columns are supplied through configuration or audit inputs; this template does not invent undocumented MetroPT columns.

## Unresolved Risks

- Failure events may still cross split boundaries unless event intervals are audited.
- Dense adjacent samples may remain temporally dependent even after the configured gap.
- Schema validation cannot prove physical sensor correctness; it only checks declared structural assumptions.
