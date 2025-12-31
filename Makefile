PYTHON ?= python

install:
	$(PYTHON) -m pip install . --no-build-isolation --force-reinstall

clean:
	rm -rf build *.egg-info dist
	find . -name "*.so" -type f -delete
	find . -name "*.pyd" -type f -delete

.PHONY: clean install
