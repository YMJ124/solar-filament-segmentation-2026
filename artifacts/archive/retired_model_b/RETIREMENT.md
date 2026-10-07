# Retired Model B experiment

Model B was retired after the matched-control phase-4 evaluation.  The best
`A:B = 0.90:0.10` blend reached PQ `0.145713228686`, while the same
`threshold=0.300`, `min_area=200` post-processing applied to Model A alone
reached PQ `0.144523554325`.  The attributable gain was `0.001189674360`,
below the predeclared `0.005` promotion gate.

The small reproducibility records in this directory are retained.  The large
checkpoints and OOF arrays were intentionally removed after their hashes were
recorded:

- `best.pt`: `fd52a10029abab578720592254a66938b3efbdaabbf460935be17015593b84f2`
- `last.pt`: `d7f8c854d1b0fa41b54cbdb0cd7b2858938b33b6ea77ae78e6317bdee5e76873`
- resolved config: `a1d01bc4d47201f8bd7baed452339f26ae8f3bf25a91b63cbc08ea6460fb5da6`
- run manifest: `413d0558ed663faafe39a9e0ff96c984addf1c967bde84e8a85843fef9b401ef`
- OOF manifest: `ae7e25ef5bcab0958f2eb9d56e8ec87a4fc9bc7b88bed7c129c5636109957b50`
- OOF NPZ aggregate: `eff3a548685fbc6e829bd136edaa3c0cb657124dd182347e3bb532084ea7a877`
- independent validation: `724ed0366d13358a341c3387b199b7fc7ee39595c32f01d5f0cbd6479e1388c3`

This retirement record is historical evidence only.  It is not an active
training or inference dependency.
