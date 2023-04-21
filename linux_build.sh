#!/bin/sh

make cuda_kernel
pip install dist/*.whl
python -m unittest test/nearest_neighbours_test.py