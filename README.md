# PyMasVis

PyMasVis is a reimplementation of [MasVis](http://www.lts.a.se/lts/masvis) in Python.

## Installation

## Usage

Analyze a file by running `poetry run pymasvis filename [filename ...]`. The result will be output to `filename-pymasvis.png`.

![Example result](docs/Rick Astley - Never Gonna Give You Up.spotify-pymasvis.png)

More options are available by running `poetry run pymasvis -h`.

## Requirements

- Python 3.8+
- [Poetry](https://python-poetry.org/)
- [FFmpeg](https://ffmpeg.org/)

## Installation

Clone the repository and run `poetry install --no-root`

## Notes

PyMasVis suppports all files that FFmpeg supports as PyMasVis uses FFmpeg to convert the file to raw PCM data before analysis.

Histogram is calculated using a maximum of 2^18 bins regardless of real bit depth. The "bits" result is scaled to the presumed real bit depth.
