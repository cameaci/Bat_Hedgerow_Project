# bats_bankable_england_v2

England-first bankable GIS screening framework for `hedge-features`.

- The output score is a planning prioritisation score for detector placement support.
- It is not a calibrated probability of bat presence.
- The framework is aligned to `bats_bankable_england_v2`.
- Temporal deployment metrics are intentionally excluded from planning-time prioritisation.
- The duplicate `mhb_roost_proxy_score` alias is excluded from the predictor registry.
- Species calibration artefacts can be added under `species_models/`, but interim production planning relies on the planner evidence engine for species and guild scenarios.
- The paired bankable enrichment profile adds lightweight categorical landscape metrics for selected WorldCover classes (`tree`, `water`, `wetland`): edge density, largest patch index, and core-area proportion. These are intended to capture habitat fragmentation and edge/context signals beyond simple buffer composition.
