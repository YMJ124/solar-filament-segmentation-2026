# P7-H1 failure review: 384 crop resolution

Date: 2026-10-18

## Local result

P7-H1 changed only the refiner crop raster from 256 to 384. The same Fold0
panel, detector candidates, C2 boundary-support policy, threshold, loss,
architecture, context scale and 12-epoch budget were retained.

The correct preregistered comparison is:

| Variant | PQ | TP pairs | FP | FN | Output instances |
|---|---:|---:|---:|---:|---:|
| 256-C2 control | 0.373099096 | 189 | 130 | 170 | 211 |
| 384-C2 candidate | 0.373960570 | 191 | 128 | 168 | 211 |
| Delta | +0.000861474 | +2 | -2 | -2 | 0 |

The candidate improved 13 images, worsened 18 and left one unchanged. It
missed the locked `+0.010` gate by `0.009138526`, so Fold1/2 training is not
authorized. Eight source instances used the reversible minimum-area fallback;
the detector source slots and upstream caches remain intact.

The evaluator's raw report says `passed_panel` because its historical built-in
comparison is against the unrefined P4 mask (`PQ=0.349214495`). That is useful
as a health check but is not the P7 decision surface. The authoritative gate is
`artifacts/instance/stage_p7/stage_report.json`.

## External comparison

Search date: 2026-10-18.

Queries included:

- `Solar Filament Segmentation 2026 Crop U-Net 256`
- `thin object segmentation crop resolution 256 boundary loss high resolution`
- `solar filament segmentation small target recall high resolution refinement`

Relevant evidence:

1. He (2026), public two-stage competition repository and paper:
   <https://github.com/HeShen-1/filament-segmentation-2026> and
   <https://github.com/HeShen-1/filament-segmentation-2026/blob/main/paper/paper.tex>.
   The reported system uses a 256 crop refiner, four flip views, five-fold OOF
   evaluation and instance-level fusion; it reports 0.3593 five-fold mean PQ
   and 0.35 public leaderboard. Its analysis argues that mask refinement is
   bounded by annotator disagreement and that fold-specific improvements need
   paired multi-fold confirmation. This directly agrees with the marginal
   `+0.00086` obtained by resolution-only scaling here. Differences: its
   detector is YOLO11m trained longer, its refiner is seed-conditioned
   ResNet18 U-Net with hard negatives, and its fusion includes a diverse public
   member, so its absolute scores are not directly transferable.

2. Shelamanov (2026), public experiment history:
   <https://www.shelamanov.com/portfolio/solar>. The reported public score rose
   from 0.35 to 0.40 through all-annotator mask training, a learned quality
   scorer/fragment merge, five-detector voting and rank-averaged scores. It does
   not publish reproducible code at the review date, so the numbers are treated
   only as directional evidence. The useful common signal is that gains came
   from recognition/selection/fusion rather than larger refiner rasters.

3. Liew et al., *Deep Interactive Thin Object Selection*, WACV 2021:
   <https://openaccess.thecvf.com/content/WACV2021/papers/Liew_Deep_Interactive_Thin_Object_Selection_WACV_2021_paper.pdf>.
   It separates high-resolution boundary processing from semantic context for
   thin structures. P5-C2 already performs boundary-local refinement while
   preserving coarse support; P7 shows that merely increasing raster size,
   without a new context/boundary representation, is insufficient here.

## Decision and next branch

P7-H1 is retained as a reversible negative result, not deleted, because the
checkpoint may be useful in a future diverse ensemble. It does not replace
P5-C2 and Fold1/2 384 models will not be trained.

The next high-probability submission branch should reuse the already validated
P5 three-fold models and test transform/model agreement while retaining every
member's masks, scores, fold, transform and checkpoint provenance. Consensus
is a final-output branch only; it must not overwrite the single-model caches.
Because current OOF assets provide only one non-leaky model per image, any
three-model test consensus must be labelled as leaderboard-calibration risk,
not as a clean OOF improvement.
