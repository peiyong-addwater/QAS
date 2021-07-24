__version__ = "0.0.1"
__author__ = "Peiyong Wang"

import numpy
numpy.core.arrayprint._line_width = 80*2
numpy.set_printoptions(precision=4)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
