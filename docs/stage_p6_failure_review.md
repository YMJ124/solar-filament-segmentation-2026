# P6 scored failure review

## Local and public evidence

Review date: 2026-10-18.

P6 was a locked full-data final fit of the already promoted T1 detector and C2
refiner.  It changed neither architecture nor post-processing.  The user
reported a rounded public PQ of `0.29`, versus approximately `0.31` for the P5
Fold0 submission.  P6 therefore failed and must not replace P5-C2.

The strongest observable shift is prediction density:

- P5: 1240 instances on 180 images, or 6.89/image;
- P6: 1463 instances on 180 images, or 8.13/image (`+18.0%`);
- P6 raw candidates increased only from 8105 to 8437 (`+4.1%`), while selected
  source instances increased by 225 before the P5 coverage fallback is counted;
- only 1011/1240 P5 instances have a P6 counterpart above IoU 0.5;
- P6 has 452 novel or materially shifted masks on 160/180 images.

Thus P6 is not merely P5 plus 223 extra small detections.  The full-data fit
changed score ordering and mask geometry across most test images.  Because the
P6 detector and refiner both trained on all legal data, there is no independent
holdout with which to recalibrate that new score distribution.  Its training-set
mAP cannot substitute for OOF evidence.

As a rough scale check only, adding 223 pure false positives to the P5 OOF
counts while holding its TP, FN, and IoU numerator fixed would reduce PQ by
roughly 0.007.  The reported public reduction is about 0.02, although both
public scores are rounded and test GT is unavailable.  This supports the view
that extra FP alone is insufficient: lost matches and changed mask IoU are also
plausible contributors.  This calculation is not a reconstruction of the
hidden score.

## External comparison

Search descriptions:

- `Solar Filament Segmentation more predictions lower score PQ metric update`
- `object detection confidence calibration distribution shift threshold`
- `post-hoc object detector calibration holdout validation set`
- `solar filament skeleton recall loss precision Dice`

The competition community previously demonstrated that the old evaluator could
reward withholding predictions.  The host later stated that the evaluation was
updated, and the current official page/self-evaluation uses Panoptic Quality:

- <https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/729809>
- <https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/730916>
- <https://www.kaggle.com/competitions/filament-segmentation-2026/overview/leaderboard-ranking>

The old metric behavior therefore must not be used as a direct explanation for
P6.  Under current PQ, however, an unmatched prediction contributes `0.5 FP` to
the denominator, so an 18% density increase is genuinely risky unless it adds
enough IoU>0.5 matches.

Object-detection calibration studies report that detector confidence is often
poorly calibrated, is affected by distribution shift and post-processing, and
that standard post-hoc calibration normally requires a separate holdout set:

- Küppers et al., *Multivariate Confidence Calibration for Object Detection*,
  CVPR Workshops 2020:
  <https://openaccess.thecvf.com/content_CVPRW_2020/html/w20/Kuppers_Multivariate_Confidence_Calibration_for_Object_Detection_CVPRW_2020_paper.html>
- Munir et al., *Towards Improving Calibration in Object Detection Under Domain
  Shift*, NeurIPS 2022:
  <https://proceedings.neurips.cc/paper_files/paper/2022/file/fcd812a51b8f8d05cfea22e3c9c4b369-Paper-Conference.pdf>
- Huseljic et al., *Systematic Evaluation of Uncertainty Calibration in
  Pretrained Object Detectors*, IJCV 2024:
  <https://link.springer.com/article/10.1007/s11263-024-02219-z>

These works do not study YOLO26, solar filaments, or competition PQ directly.
They support only the mechanism-level conclusion that the P5 threshold cannot
be assumed to retain the same meaning after P6 retraining.  They do not justify
choosing a new P6 threshold from one leaderboard score.

A recent solar-filament preprint reports that ordinary Skeleton Recall Loss did
not materially improve its manual-spine coverage, while a spine-anchored variant
increased spine recall but reduced Dice, clDice, and precision:
<https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7441885>.  It is unusually
domain-relevant but is a non-peer-reviewed preprint with a different plain U-Net
and evaluation target.  Given P6's current precision warning, it argues against
making topology-recall loss the immediate next variable.

## Decision and next controlled candidate

1. P6 is rejected as a submission branch.  Keep its checkpoint, configuration,
   audit, disagreement report, and receipt as negative evidence; do not delete
   the upstream predictions or overwrite P5.
2. Do not search P6 confidence, area, cap, or refiner threshold against its one
   public score.  There is no valid holdout for such calibration.
3. Return to the OOF-validated P5/T1 detector and preserve its instance set.
4. The next single-variable candidate is P7-H1: increase only the local refiner
   crop resolution from 256 to 384 while keeping the architecture, BCE+Dice+
   boundary loss, detector candidates, 0.5 mask threshold, and reversible C2
   support policy fixed.
5. First compare 256 and 384 C2 on the same locked Fold0 32-image panel.  P7-H1
   must gain at least `+0.010 PQ` over the 256 C2 panel without changing source
   instance count.  If it fails, retire it without training Fold1/2.

P7-H1 targets boundary/mask precision rather than candidate count.  It is fully
reversible and leaves T1 caches, source IDs, coarse masks, confidence scores,
and P5 checkpoints intact.
