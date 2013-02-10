FILES = src/PyMasVis.py src/pymasvis/analyze.py src/pymasvis/gui.py src/pymasvis/widgets.py
OPTIONS = --iconfile res/Cupcake.icns --resources res/ffmpeg

devel: ${FILES}
	python setup.py py2app -A ${OPTIONS}

dist: clean ${FILES}
	python setup.py py2app ${OPTIONS}

	chmod +x "dist/PyMasVis.app/Contents/Resources/ffmpeg"

	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/matplotlib/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/numpy/core/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/numpy/distutils"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/numpy/f2py/docs"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/numpy/f2py/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/numpy/fft/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/numpy/lib/benchmarks"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/numpy/lib/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/numpy/linalg/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/numpy/ma/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/numpy/matrixlib/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/numpy/oldnumeric/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/numpy/polynomial/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/numpy/random/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/numpy/testing/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/numpy/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/scipy/cluster/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/scipy/constants/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/scipy/fftpack/benchmarks"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/scipy/fftpack/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/scipy/integrate/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/scipy/interpolate/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/scipy/io/arff/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/scipy/io/harwell_boeing/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/scipy/io/matlab/benchmarks"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/scipy/io/matlab/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/scipy/io/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/scipy/lib/blas/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/scipy/lib/lapack/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/scipy/linalg/benchmarks"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/scipy/linalg/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/scipy/misc/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/scipy/ndimage/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/scipy/odr/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/scipy/optimize/benchmarks"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/scipy/optimize/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/scipy/signal/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/scipy/sparse/benchmarks"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/scipy/sparse/csgraph/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/scipy/sparse/linalg/dsolve/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/scipy/sparse/linalg/dsolve/umfpack/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/scipy/sparse/linalg/eigen/arpack/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/scipy/sparse/linalg/eigen/lobpcg/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/scipy/sparse/linalg/isolve/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/scipy/sparse/linalg/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/scipy/sparse/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/scipy/spatial/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/scipy/special/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/scipy/stats/tests"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/scipy/weave/doc"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/scipy/weave/examples"
	rm -rf "dist/PyMasVis.app/Contents/Resources/lib/python2.7/scipy/weave/tests"

clean:
	rm -rf build dist

.PHONY: clean
