import numpy as np
import pytest
from PIL import Image

from pymasvis.main import run


def test_detailed(detailed_reference_image, test_wav_path, tmp_path):
    test_image_path = tmp_path / 'test.wav-pymasvis.png'
    run(test_wav_path, destdir=tmp_path)

    test_image = Image.open(test_image_path)

    ref = np.asarray(detailed_reference_image)
    res = np.asarray(test_image)

    print(ref, res)

    err = np.sum((ref.astype(float) - res.astype(float)) ** 2)
    print(err)
    err /= float(ref.shape[0] * ref.shape[1])
    print(err)

    assert err < 0.01
