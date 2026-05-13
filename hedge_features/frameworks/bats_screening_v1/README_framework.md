# bats_screening_v1

GIS-only hedgerow bat survey prioritisation framework for `hedge-features`.

- Output score is `survey_priority_score` (ranking / prioritisation score), not a guaranteed probability.
- Default policy is `Recall-first`.
- Confidence and reason codes are first-class outputs and can override automated deprioritisation.
- Strict GIS-only predictor registry excludes known leakage / non-transfer columns.

Artefacts in this package are versioned and loaded at runtime by the Streamlit screening UI and screening engine.

Species-target calibration artefacts can be added under `species_models/` using the `hedge-features train-species-model` command.

When present, the screening engine can append species-specific probabilities plus domain-of-applicability outputs for those trained targets.
