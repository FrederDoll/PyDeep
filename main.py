
import numpy as np
#import torch

# Press the green button in the gutter to run the script.
import torchphysics as tp

if __name__ == '__main__':
    print('This is the first try of torchphysiks')
    print('https://torchphysics.readthedocs.io/en/latest/tutorial/solve_pde.html')
    # we need to define so called spaces for input and Output variables
    X = tp.spaces.R2('x')
    T = tp.spaces.R1('t')
    G = X * T
    print(G)
    print("heisa hossa (Done)")
