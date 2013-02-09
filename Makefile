FILES = src/PyMasVis.py src/pymasvis/analyze.py src/pymasvis/gui.py src/pymasvis/widgets.py
OPTIONS = --iconfile res/Cupcake.icns --resources res/ffmpeg

devel: ${FILES}
	python setup.py py2app -A ${OPTIONS}

dist: ${FILES}
	python setup.py py2app ${OPTIONS}

clean:
	rm -rf build dist

.PHONY: clean
