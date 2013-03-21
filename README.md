# PyMasVis

PyMasVis is a reimplementation of [MasVis](http://www.lts.a.se/lts/masvis) in Python.

## Requirements

- NumPy
- SciPy
- Matplotlib
- pyspotify

FFmpeg is required for anything other than WAV files.

### Requirement installation

	pip install numpy
	pip install scipy
	pip install matplotlib
	pip install pyspotify

scipy requires ATLAS/BLAS/LAPACK. Fortran compiler (gfortran).

PySpotify requires separate installation of libspotify. Libspotify can be downloaded from https://developer.spotify.com/technologies/libspotify/

On Linux the contents of lib/ goes in /usr/local/lib and include/ in /usr/local/include

## Usage

Analyse a file by running `python src/pymasvis/analyze.py filename`. The result will be output to `filename-pymasvis.png`. Spotify analysis is done by running `python src/pymasvis/analyze.py -u username -p password spotify_link`. Result when using Spotify is output to current working directory.

## Notes

PyMasVis supports WAV files. MP3 and other formats are supported if FFmpeg is installed.

Histogram is calculated using a maximum of 2^16 bins regardless of real bit depth. The "bits" result is scaled to the presumed real bit depth.

## Binary

Application bundle for Mac OS X 10.8 can be found here: http://albin.abo.fi/user/jfors/pub/PyMasVis.app.zip NOTE: GUI under development, very buggy. ;)

