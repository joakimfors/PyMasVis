# PyMasVis

PyMasVis is a reimplementation of [MasVis](http://www.lts.a.se/lts/masvis) in Python.

## Requirements

- NumPy
- SciPy
- Matplotlib

FFmpeg is required for anything other than WAV files.

## Usage

Analyse a file by running `python src/pymasvis/analyze.py filename`. The result will be output to `filename-pymasvis.png`.

## Notes

PyMasVis supports WAV files. MP3 and other formats are supported if FFmpeg is installed.

Histogram is calculated using a maximum of 2^16 bins regardless of real bit depth. The "bits" result is scaled to the presumed real bit depth.

## Binary

Application bundle for Mac OS X 10.8 can be found here: http://albin.abo.fi/user/jfors/pub/PyMasVis.app.zip NOTE: GUI under development, very buggy. ;)

