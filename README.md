# PyMasVis

PyMasVis is a reimplementation of [MasVis](http://www.lts.a.se/lts/masvis) in Python.

## Requirements

- NumPy
- SciPy
- Matplotlib

FFmpeg is required for anything other than WAV files.

## Usage

Analyse a file by running `python pymasvis.py filename`. The result will be output to `filename.png`.

## Notes

PyMasVis supports libsndfile compatible formats (wav, flac, au, ogg etc). MP3 and other formats are supported if FFmpeg is installed.

Histogram is calculated using 2^16 bins regardless of real bit depth. The "bits" result is scaled to the presumed real bit depth.

