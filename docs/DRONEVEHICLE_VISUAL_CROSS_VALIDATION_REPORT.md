# DroneVehicle RGB/IR Cross-Validation Report

This report adds the same leakage-controlled evaluation discipline used for the acoustic benchmark to the DroneVehicle RGB and infrared vehicle layers.

## Protocol

- Final lockbox: 20% of each modality label file is written to a manifest and not evaluated.
- Development evaluation: stratified 5-fold cross-validation on a capped 500-image development set.
- Tuning/evaluation separation: each fold writes a tuning manifest containing the other folds and evaluates only the held-out fold.
- Leakage check: every fold reported zero overlap between tuning and evaluation examples.
- API usage: none. These runs use the local vehicle proposal layer only.

The lockbox manifests are:

```text
logs/visual_cross_validation/dronevehicle_rgb/final_test_lockbox.csv
logs/visual_cross_validation/dronevehicle_ir/final_test_lockbox.csv
```

Do not evaluate those lockbox files during tuning. They should be used exactly once after the vehicle proposal/review policy is frozen.

## Results

| Modality | Development Images | Lockbox Images | Capture Precision | Capture Recall | Capture F1 |
|---|---:|---:|---:|---:|---:|
| RGB vehicle local CV | 500 | 5,688 | 50.0% +/- 0.0% | 100.0% +/- 0.0% | 66.7% +/- 0.0% |
| IR vehicle local CV | 500 | 5,688 | 91.6% +/- 0.5% | 100.0% +/- 0.0% | 95.6% +/- 0.3% |

## Interpretation

The RGB local proposal layer is intentionally high-recall and noisy. It captured every positive vehicle frame in the development folds, but it also captured every negative frame, producing 50.0% capture precision on the balanced RGB development sample. This confirms that RGB still benefits from selective API cleanup or a stricter local review policy.

The infrared local proposal layer remains stronger for this dataset. It preserved all positive vehicle evidence and reached 91.6% mean capture precision on the capped development folds. The IR development set has fewer negatives because the source IR labels contain very few negative frames, so the lockbox should remain untouched until the policy is frozen.

## Anti-Leakage Note

The earlier single-subset local RGB/IR baselines are still useful smoke tests, but these cross-validation runs are better for development evaluation because every capped development image gets exactly one held-out evaluation pass and no image appears in both the tuning and evaluation split for the same fold.
