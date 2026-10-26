# P8 result review and submission decision

Date: 2026-10-18

## Full inference health

P8-E1 completed all 180 test images on the RTX 5070 Laptop GPU in 3302.57 s
(55.0 minutes). It wrote 540 per-member image caches for three folds and four
transforms, covering 14,703 refined member candidates. CUDA peak allocation was
1.56 GiB and peak RSS was 4.61 GiB. All source hashes, 180-image coverage,
compressed RLEs and non-overlap checks passed.

## Why raw support>=2 is not submitted

The raw support>=2 output contains 1609 instances, 369 more than P5 (+29.8%)
and 146 more than the failed P6 submission. Relative to P5, 462 masks are new
or shifted below IoU 0.5; 334 of those have area below 1600 pixels. This nearly
repeats P6's failure signature (452 new/shifted masks, 306 below 1600), despite
the new masks having cross-fold support. P8-E1 is therefore retained as an
auditable research branch but is not recommended for submission.

## Conservative offline variants

All variants were assembled from immutable completed caches without new GPU
inference:

| Variant | Instances | P5 preservation | New instances | Decision |
|---|---:|---:|---:|---|
| E2: three-fold unanimous only | 1276 | 1042/1240 at IoU>0.5 | 234 new/shifted | Reject: loses/replaces 198 P5 instances |
| E3: P5 + unanimous novel | 1454 | 1240/1240 exact | 214 | Too close to P6 density |
| E4: P5 + unanimous novel + P6 agreement | 1368 | 1240/1240 exact | 128 | Recommended challenger |

E4 adds only instances that are absent from P5, supported by all three P5 fold
models, and also matched by the independently trained P6 full-fit model at IoU
greater than 0.5. It rejects 86 unanimous-fold additions lacking P6 support.
The 128 accepted additions occur on 93 images; 87 images remain byte-equivalent
to their P5 masks. No P5 mask is removed, reshaped or overwritten.

E4 passes an independent submission audit: 1368 rows, 180/180 images, zero
missing images, 600 minimum area, maximum 17 instances per image, valid RLEs
and no overlap. SHA256:
`c30b793bc43d845a55eb30e95bd4560bb269d1158a151e8c0b51de6d36e19927`.

## External comparison

Search date: 2026-10-18.

Queries:

- `filament segmentation fusion same-family correlated false positives support 2`
- `five detector ensemble rank averaged scores solar filaments`
- `instance segmentation ensemble consensus false positives same family models IoU clustering`

Evidence:

1. He (2026) reports that expanding same-family YOLO ensembles to seven members
   reduced the leaderboard score by 0.01 because correlated false positives
   gained votes. Its successful three-member fusion uses IoU>=0.5 and support>=2,
   but includes a distinct external Mask R-CNN member. Source:
   <https://github.com/HeShen-1/filament-segmentation-2026/blob/main/paper/paper.tex>.
   This directly argues against submitting P8-E1's 1609-instance same-family
   output. E4 modifies the idea by treating P5 as immutable and requiring an
   additional full-fit model only as an admission gate for novel instances.

2. Shelamanov's public experiment history reports 0.39 with a five-detector
   ensemble and 0.40 after rank averaging, but also records score drops when
   simply keeping more or fewer masks. Source:
   <https://www.shelamanov.com/portfolio/solar>. This supports model agreement
   as a useful signal while warning that instance-count changes alone are not
   monotonic improvements. The implementation is not public at the review date,
   so it is directional rather than reproducible evidence.

3. The general high-recall ensemble rationale is that true positives tend to
   be consistent while false positives differ across diverse members. The key
   limitation here is diversity: P5 folds and P6 share the same model family,
   so four-model agreement is stronger evidence than two-fold support but not
   independent proof. This is why E4 remains a leaderboard-calibration
   challenger rather than replacing the P5 research baseline.

## Official score and final decision

E4 was submitted and the user reported a rounded public score of `0.31`. P5
was also reported as approximately `0.31`, so the receipt cannot resolve a
possible sub-0.01 difference, but it shows no visible gain. The 128 additions
therefore produced approximately zero net leaderboard benefit despite
three-fold unanimity and P6 agreement. E4 is `scored_not_promoted`; P5-C2
remains the default.

`artifacts/submissions/p8_e4_p5_plus_threefold_unanimous_p6_agreement_1536.csv`

Receipt:
`artifacts/submissions/p8_e4_p5_plus_threefold_unanimous_p6_agreement_1536.receipt.json`.

This closes threshold/support tuning within the same detector family. The next
stage must return to clean OOF evidence. P5 already trains its refiner on every
available annotator batch with equal total sampling mass per physical image,
so that public idea is not a new variable here. The next diagnostic is instead
an OOF fragment-merge upper bound, followed only if justified by an OOF-trained
set-level quality/merge model. Raw masks and source slots remain immutable; no
merge may destructively replace them.
