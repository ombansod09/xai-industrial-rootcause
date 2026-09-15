# Research Protocol

- Chronological evaluation only.
- Fit scalers/thresholds only on training/validation data.
- Prevent windows from crossing train/test boundaries.
- Fixed seeds and fixed test set.
- Report negative findings, not only wins.
- Feature attribution is model attribution, not automatically physical causation.

Robustness tests:
1. sensor noise
2. sensor dropout
3. missing blocks
4. temporal jitter
5. feature permutation
6. correlated-feature stress
7. distribution shift
8. random-seed variation
9. XAI-method variation
10. counterfactual feasibility

Root-cause claims must be separated into model attribution, anomaly detection, candidate ranking, and maintenance/physical evidence.
