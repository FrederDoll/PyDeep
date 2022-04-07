
import numpy as np
import torch
import torchphysics as tp
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('This is the first try of torchphysiks')
    print('https://torchphysics.readthedocs.io/en/latest/tutorial/solve_pde.html')
    # we need to define so called spaces for input and Output variables
    X = tp.spaces.R2('x')  # input is 2D
    U = tp.spaces.R1('u')  # output is 1D

    #  we need a Domain
    #square = tp.domains.Parallelogram(X, [0, 0], [1, 0], [0, 1])