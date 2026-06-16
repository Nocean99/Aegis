## Aegis Acoustic Benchmark v1

Dataset:

- Anthropogenic: 77
- Animal: 265
- Sonar: 26

Results:

- Development CV precision: 34.2% ± 8.9%
- Development CV recall: 70.6% ± 16.7%
- Development CV F1: 46.0% ± 11.5%
- Final lockbox: 73 clips set aside, not evaluated

Key Finding: The current vessel-aware acoustic gate is useful as a recall-oriented triage layer, but it is still noisy on animal and sonar negatives. The final test lockbox should remain untouched until thresholds or a learned acoustic model are frozen.
