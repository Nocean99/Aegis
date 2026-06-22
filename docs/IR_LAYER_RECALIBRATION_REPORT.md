# IR Vehicle Layer Recalibration Report

## Goal

Improve the infrared vehicle layer's ability to separate true vehicles from thermal false positives without tuning against the 40-image video demo set.

The demo set was used only for diagnosis. The actual before/after comparison used the leakage-controlled DroneVehicle IR cross-validation harness with a protected lockbox held out.

## Diagnosis

The high-priority false positives in the mixed IR demo were not empty frames. They were warm or high-contrast thermal structures such as pavement, road markings, lots, buildings, and edge artifacts.

Several negative images received:

- detector confidence: `0.860`
- proposal reason: `hot IR blob`
- semantic decisions: `LIKELY_MATCH` or `POSSIBLE_MATCH`
- review priority: `0.72-0.95`

The failure mode was therefore not missing evidence. It was over-promoting generic hot blobs into confirmed vehicle matches.

## Change

The local semantic triage now uses stricter decision thresholds for infrared vehicle proposals and hot-blob proposals:

- RGB vehicle proposals keep the existing thresholds.
- IR / `hot IR blob` proposals require stronger evidence before becoming `POSSIBLE_MATCH` or `LIKELY_MATCH`.
- Weak thermal blobs remain `NEEDS_REVIEW`, preserving the recall-biased workflow without treating every thermal blob as confirmed vehicle evidence.

This is a conservative scoring recalibration, not a trained thermal detector.

## Held-Out Cross-Validation Results

Dataset: DroneVehicle IR benchmark  
Evaluation: 5-fold development cross-validation with final lockbox held out  
Development cap: 500 images  
API: not used

| Metric | Before | After |
|---|---:|---:|
| Confirmed precision | 91.8% +/- 0.7% | 92.5% +/- 1.0% |
| Confirmed recall | 100.0% +/- 0.0% | 99.6% +/- 0.5% |
| Confirmed F1 | 95.7% +/- 0.4% | 95.9% +/- 0.7% |
| Capture precision | 91.6% +/- 0.4% | 91.6% +/- 0.4% |
| Capture recall | 100.0% +/- 0.0% | 100.0% +/- 0.0% |
| Capture F1 | 95.6% +/- 0.2% | 95.6% +/- 0.2% |

## Interpretation

The recalibration slightly improves confirmed-match discrimination, but it does not solve the core IR false-positive problem. Raw hot-blob confidence overlaps heavily between true vehicles and negatives, so thresholding alone has limited upside.

The important result is that uncertain IR evidence is still captured for review, while weaker thermal blobs are less likely to be presented as confirmed vehicle matches.

## Semantic Layer Note

Local CLIP was not available in this environment, so it was not measured for this pass. API review should be tested only on a small held-out IR sample. Thermal imagery may not benefit from RGB-trained vision models without stricter prompts or thermal-specific data, so that should be measured rather than assumed.

## Future Work

- Train or fine-tune a thermal vehicle detector.
- Add thermal-specific hard negatives: warm pavement, roofs, road markings, industrial lots, and edge artifacts.
- Evaluate a stricter API thermal prompt on a small held-out sample.
- Keep final lockbox sealed until the IR configuration is frozen.
