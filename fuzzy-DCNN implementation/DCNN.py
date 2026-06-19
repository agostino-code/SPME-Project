ACTIVATION = 'relu'
FINAL_ACTIVATION = 'linear'
NUM_FILTERS = 32
POOL_SIZE = 2
OPTIMIZER = 'adam'
LOSS_FUNCTION = 'mean_squared_error'

"""
For each sample, compute an m x km x k matrix:

    rows = features,

    columns = cluster membership responses.

Feed that matrix to the CNN regressor.

Output one scalar effort value.
"""

class DCNN:
    def __init__(self, input, output_shape):
        self.input = input
        self.input_shape = input.shape
        self.output_shape = output_shape
