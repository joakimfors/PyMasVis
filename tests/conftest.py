from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def fixture_data_dir() -> Path:
    return Path(__file__).parent / 'data'


@pytest.fixture
def test_wav_path(fixture_data_dir):
    return fixture_data_dir / 'test.wav'


@pytest.fixture
def detailed_reference_image(fixture_data_dir: Path) -> Image:
    image_path: Path = fixture_data_dir / 'test.wav-pymasvis.png'
    image: Image = Image.open(image_path)
    return image
