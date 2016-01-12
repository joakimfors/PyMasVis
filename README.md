# PyMasVis

PyMasVis is a reimplementation of [MasVis](http://www.lts.a.se/lts/masvis) in Python.


## Usage

Analyze a file by running `python src/analyze.py filename`. The result will be output to `filename-pymasvis.png`. Spotify analysis is done by running `python src/analyze.py -u username -p password spotify_uri`. Result when using Spotify is output to the current working directory.

![Example result](doc/Rick Astley - Never Gonna Give You Up.spotify-pymasvis.png)


## Requirements

- NumPy
- SciPy
- Matplotlib
- Pillow or PIL
- (pyspotify)

FFmpeg is required for anything other than WAV files.


### Requirement installation

For standard usage on .wav files:

	pip install numpy
	pip install scipy
	pip install matplotlib
	pip install pillow

To analyze spotify URIs:

	pip install pyspotify


## Notes

PyMasVis supports WAV files. MP3 and other formats are supported if FFmpeg is installed.

Histogram is calculated using a maximum of 2^16 bins regardless of real bit depth. The "bits" result is scaled to the presumed real bit depth.


## Troubleshooting

Scipy requires ATLAS/BLAS/LAPACK. Fortran compiler (gfortran).

PySpotify might require a separate installation of libspotify. Libspotify can be downloaded from https://developer.spotify.com/technologies/libspotify/

On Linux the contents of lib/ goes in /usr/local/lib and include/ in /usr/local/include

On OS X libspotify.framework/Versions/XX.XX.XX/Headers/api.h goes into /usr/include/libspotify and libspotify.framework/libspotify is copied to /usr/lib/libspotify.so
