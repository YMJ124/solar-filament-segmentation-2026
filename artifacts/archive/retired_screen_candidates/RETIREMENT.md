# Retired screening candidates

Retired on 2026-10-07 after evaluation on the fixed 32-image Fold-0 screen.
`resnet34_256` remains the semantic control at PQ `0.122442`.

| Candidate | Best PQ | Best epoch | Decision |
|---|---:|---:|---|
| `resnet101_256` | 0.116156 | 3 | Lower PQ, slower, and larger than the control |
| `resnet101_512` | 0.009173 | 3 | Failed the fast screen |
| `resnet34_512` | 0.000253 | 3 | Failed the fast screen |
| `resnet34_full512_balanced` | 0.004217 | 3 | Full-frame downsampling lost thin structures |
| `flatunet_c32_full512_balanced` | 0.013501 | 1 | Severe fragmentation and false positives |

The last two checkpoints were also rescored with a small diagnostic grid. The
best ResNet34 full-frame setting reached PQ `0.040180` at threshold `0.70`,
minimum area `1000`. The best Flat U-Net setting reached PQ `0.066505` at
threshold `0.40`, minimum area `500`. Both remained below the control.

The `runs/` directory retains resolved configs, manifests, overfit checks and
every epoch report. Model weights were removed because they are large and fully
reconstructable. Their hashes are retained below.

## Removed checkpoint hashes

| Candidate | File | SHA-256 |
|---|---|---|
| `resnet101_256` | `best.pt` | `0f4233b08d217af0feeea676a918aa07cecc38c69f1e35d2d497332f4928ca4e` |
| `resnet101_256` | `last.pt` | `b051fd043bc9d40113cd6b0f3f7080b6a67ce909eb9abc40b5d10332ca45a528` |
| `resnet101_512` | `best.pt` | `82f041efe51077924aa7831de53918b39142c7f3b5eb411f2be5f5d9cf224980` |
| `resnet101_512` | `last.pt` | `c26edb30339d033b3bc70aa661ebe845c79e5af0c78eaddb97e14fc3c5165802` |
| `resnet34_512` | `best.pt` | `be3c7fde35f813470ba29d05d36de66c2b85c363f9b4fdf3e0a89e0829da29a5` |
| `resnet34_512` | `last.pt` | `76e34a07fd6cac67ace0f14a191fc29f1b21e0f31e61f11b4bb59bfa0c40d51c` |
| `resnet34_full512_balanced` | `best.pt` | `2a2bd36edcf227e0e965b5c14af549ed1d5e5d14330857f92bd347a9b4cf841e` |
| `resnet34_full512_balanced` | `last.pt` | `e6a82e39a7259a044e35640869043dc878b42ec961caeebc2c8b579b5e95dc7a` |
| `flatunet_c32_full512_balanced` | `best.pt` | `e8c82ba920576b73d3ae1301dc6b4ac7a47accc4e2cc67f0b689a8dbf93b6c5f` |
| `flatunet_c32_full512_balanced` | `last.pt` | `b3d259e03a7207103f223f829858dc2c1d0be188bee0a57b4d42eb11131f6710` |
