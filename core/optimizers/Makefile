PYBIND11_INCLUDES := $(shell python3 -m pybind11 --includes)

compile:
	c++ -O3 -Wall -shared -std=c++17 -fPIC $(PYBIND11_INCLUDES) linear.cpp -o mylib$(shell python3-config --extension-suffix)