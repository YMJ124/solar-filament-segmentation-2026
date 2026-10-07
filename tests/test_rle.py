import numpy as np
import pytest

from src.submission.rle import decode_rle, encode_mask

pycocotools = pytest.importorskip("pycocotools")


def test_rle_roundtrip():
    original = np.zeros((32, 32), dtype=np.uint8)
    original[3:10, 5:17] = 1
    encoded = encode_mask(original)
    decoded = decode_rle(encoded, height=32, width=32)
    np.testing.assert_array_equal(decoded, original.astype(bool))
