NVCC=nvcc
TARGET=build/add
LIBS=-lcufft

$(TARGET): add.cu
	@mkdir -p build
	$(NVCC) -o $(TARGET) add.cu $(LIBS)

install:
	python setup.py build_ext --build-dir=build install --build-lib=build

clean:
	rm -f $(TARGET) *.so *.pyd
	rm -rf build *.egg-info

.PHONY: clean install
