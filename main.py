
import numpy as np
#import torch

# Press the green button in the gutter to run the script.

import torch
#import torchphysics as tp
import math
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from fdm_heat_equation import FDM, transform_to_points
from tqdm import tqdm

import os

## let's import the relevant libraries for try2
import torch.nn as nn
from time import perf_counter
from PIL import Image
from functools import partial
import numpy as np
import requests


"""""
def firstTry():
    print('This is the first try of torchphysiks')

    # First, we create the spaces for our problem.These define the variable names which will be used in the remaining part of this code.

    X = tp.spaces.R2('x')  # x is the space variable
    T = tp.spaces.R1('t')  # t corresponds to the time
    D = tp.spaces.R1('D')  # D is an interval of diffusions
    U = tp.spaces.R1('u')  # u is the variable for the (1D-)solution.

    # Domain Creation ?
    h, w = 20, 20
    A_x = tp.domains.Parallelogram(X, [0, 0], [w, 0], [0, h])
    A_t = tp.domains.Interval(T, 0, 40)
    A_D = tp.domains.Interval(D, 0.1, 1.0)

    inner_sampler = tp.samplers.AdaptiveRejectionSampler(A_x * A_t * A_D, density=1)
    # initial values should be sampled on the left boundary of the time interval and for every x and D
    initial_v_sampler = tp.samplers.RandomUniformSampler(A_x * A_t.boundary_left * A_D, density=1)

    boundary_v_sampler = tp.samplers.RandomUniformSampler(A_x.boundary * A_t * A_D, density=1)
    # We visualize the domain through the points created by the samplers using matplotlib:
    tp.utils.scatter(X * T, inner_sampler, initial_v_sampler, boundary_v_sampler)
    plt.show()
    print("PLOT is now shown")
    # n the next step we define the NN-model we want to fit to the PDE. A normalization can improve convergence for large or small domains.

    model = tp.models.Sequential(
        tp.models.NormalizationLayer(A_x * A_t * A_D),
        tp.models.FCN(input_space=X * T * D, output_space=U, hidden=(50, 50, 50))
    )
    print("model is defined")

    # Now, we define a condition which aims to minimze the mean squared error of the residual of the poisson equation.
    def heat_residual(u, x, t, D):
        return D * tp.utils.laplacian(u, x) - tp.utils.grad(u, t)

    pde_condition = tp.conditions.PINNCondition(module=model,
                                                sampler=inner_sampler,
                                                residual_fn=heat_residual,
                                                name='pde_condition')
    print("conditions are defined")

    # Additionally, we add a boundary condition at the boundary of the domain:
    def boundary_v_residual(u):
        return u

    boundary_v_condition = tp.conditions.PINNCondition(module=model,
                                                       sampler=boundary_v_sampler,
                                                       residual_fn=boundary_v_residual,
                                                       name='boundary_condition')
    print("boaundary conditions")

    # The initial condition can be defined via a data function.
    # #Again, we minimize the mean squared error over the sampled points.
    def f(x):
        return torch.sin(math.pi / w * x[:, :1]) * torch.sin(math.pi / h * x[:, 1:])

    def initial_v_residual(u, f):
        return u - f

    initial_v_condition = tp.conditions.PINNCondition(module=model,
                                                      sampler=initial_v_sampler,
                                                      residual_fn=initial_v_residual,
                                                      data_functions={'f': f},
                                                      name='initial_condition')

    print("Initial conditions are done now")

    ##For comparison, we compute the solution via a finite difference scheme.#############

    fdm_domain, fdm_time_domains, fdm_solution = FDM([0, w, 0, h], 2 * [2e-1], [0, 5],
                                                     [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], f)
    print("we did FDM")
    fdm_inp, fdm_out = transform_to_points(fdm_domain, fdm_time_domains, fdm_solution,
                                           [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], True)
    print("Comparission as provided with torch physisk")
    # Comparsion to measured or computed data can be performed via a DataCondition using data supplied via a PointsDataLoader.
    val_condition = tp.conditions.DataCondition(module=model, dataloader=tp.utils.PointsDataLoader((fdm_inp, fdm_out),
                                                                                                   batch_size=8000),
                                                norm='inf')
    print("Caomparission computed via DataCondition")
    # Finally, we optimize the conditions using a pytorch-lightning.LightningModule Solver and running the training.
    # In the Solver, the training and validation conditions, as well as all optimizer options can be specified.
    solver = tp.solver.Solver([pde_condition, boundary_v_condition, initial_v_condition], [val_condition])

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    trainer = pl.Trainer(gpus=0,  # or None for CPU
                         max_steps=2000,
                         logger=False,
                         benchmark=True,
                         val_check_interval=400,
                         checkpoint_callback=False)
    trainer.fit(solver)

    anim_sampler = tp.samplers.AnimationSampler(A_x, A_t, 10, n_points=100, data_for_other_variables={'D': 1.0})
    anim = tp.utils.animate(model, lambda u: u[:, 0], anim_sampler, ani_speed=10)

    print("heisa hossa (Done)")

def secondTry():
    ## check if GPU is available and use it; otherwise use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    # N is a Neural Network - This is exactly the network used by Lagaris et al. 1997
    N = nn.Sequential(nn.Linear(1, 50), nn.Sigmoid(), nn.Linear(50, 1, bias=False))

    # Initial condition
    A = 0.

    # The Psi_t function
    Psi_t = lambda x: A + x * N(x)

    # The right hand side function
    f = lambda x, Psi: torch.exp(-x / 5.0) * torch.cos(x) - Psi / 5.0

    # The loss function
    def loss(x):
        x.requires_grad = True
        outputs = Psi_t(x)
        Psi_t_x = torch.autograd.grad(outputs, x, grad_outputs=torch.ones_like(outputs),
                                      create_graph=True)[0]
        return torch.mean((Psi_t_x - f(x, outputs)) ** 2)

    # Optimize (same algorithm as in Lagaris)
    optimizer = torch.optim.LBFGS(N.parameters())

    # The collocation points used by Lagaris
    x = torch.Tensor(np.linspace(0, 2, 100)[:, None])

    # Run the optimizer
    def closure():
        optimizer.zero_grad()
        l = loss(x)
        l.backward()
        return l

    for i in range(10):
        optimizer.step(closure)

    # Let's compare the result to the true solution
    xx = np.linspace(0, 2, 100)[:, None]
    with torch.no_grad():
        yy = Psi_t(torch.Tensor(xx)).numpy()
    yt = np.exp(-xx / 5.0) * np.sin(xx)

    fig, ax = plt.subplots(dpi=100)
    ax.plot(xx, yt, label='True')
    ax.plot(xx, yy, '--', label='Neural network approximation')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$\Psi(x)$')
    plt.legend(loc='best');
    plt.show()
"""
def Sine():

    print("we want to dolfe the DE: x''=-x, x(0)=0,x'(0)=1")
    print("ressource for Math https://www.emathhelp.net/en/calculators/differential-equations/differential-equation-calculator/?i=x%27%27%3D-x%2C+x%280%29%3D0%2Cx%27%280%29%3D1")
    print("this is based on: https://benmoseley.blog/my-research/so-what-is-a-physics-informed-neural-network/ ")
    print("other important source would be: https://proxy.nanohub.org/weber/2018450/0RQRgA9Vr8Pc1kMl/10/notebooks/PINNs_V1-1.ipynb?")

    def save_gif_PIL(outfile, files, fps=5, loop=0):
        "Helper function for saving GIFs"
        imgs = [Image.open(file) for file in files]
        imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000 / fps),
                     loop=loop)

    def plot_result(x, y, x_data, y_data, yh, xp=None):
        "Pretty plot training results"
        plt.figure(figsize=(8, 4))
        plt.plot(x, y, color="grey", linewidth=2, alpha=0.8, label="Exact solution")
        plt.plot(x, yh, color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
        plt.scatter(x_data, y_data, s=60, color="tab:orange", alpha=0.4, label='Training data')
        if xp is not None:
            plt.scatter(xp, -0 * torch.ones_like(xp), s=60, color="tab:green", alpha=0.4,
                        label='Physics loss training locations')
        l = plt.legend(loc=(1.01, 0.34), frameon=False, fontsize="large")
        plt.setp(l.get_texts(), color="k")
        plt.xlim(0, 40)
        plt.ylim(-1.1, 1.1)
        plt.text(1.065, 0.7, "Training step: %i" % (i + 1)+"loss:"%loss, fontsize="xx-large", color="k")
        plt.axis("off")

    class FCN(nn.Module):
        "Defines a connected network"

        def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
            super().__init__()
            activation = nn.Tanh
            self.fcs = nn.Sequential(*[
                nn.Linear(N_INPUT, N_HIDDEN),
                activation()])
            self.fch = nn.Sequential(*[
                nn.Sequential(*[
                    nn.Linear(N_HIDDEN, N_HIDDEN),
                    activation()]) for _ in range(N_LAYERS - 1)])
            self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

        def forward(self, x):
            x = self.fcs(x)
            x = self.fch(x)
            x = self.fce(x)
            return x
    ##Generate Training Data:
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("using GPU")
    else:
        device = torch.device("cpu")
        print("using CPU")

    # get the analytical solution over the full domain
    x = torch.linspace(0, 50, 10000).view(-1, 1).to(device)
    #y = oscillator(d, w0, x).view(-1, 1)
    y=torch.sin(x).view(-1, 1).to(device)
    #print(x.shape, y.shape)

    # slice out a small number of points from the LHS of the domain
    x_data = x[0:400:50]
    y_data = y[0:400:50]
    #print(x_data.shape, y_data.shape)

    plt.figure()
    plt.plot(x.cpu(), y.cpu(), label="Exact solution")
    plt.scatter(x_data.cpu(), y_data.cpu(), color="tab:orange", label="Training data")


    # sample locations over the problem domain
    x_physics = torch.linspace(0, 3*2*math.pi, 50).view(-1, 1).requires_grad_(True)

    plt.scatter(x_physics.detach().numpy(),torch.ones_like(x_physics)*0,color="tab:green", label="x_Physiks")
    plt.legend()
    plt.show()
    x_physics = x_physics.cuda()
    torch.manual_seed(123)
    model = FCN(1, 1, 32, 3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    files = []
    for i in tqdm(range(100000)):

        optimizer.zero_grad()

        # compute the "data loss"
        yh = model(x_data)
        loss1 = torch.mean((yh - y_data) ** 2)  # use mean squared error

        # compute the "physics loss"
        yhp = model(x_physics)
        dx = torch.autograd.grad(yhp, x_physics, torch.ones_like(yhp), create_graph=True)[0]  # computes dy/dx
        dx2 = torch.autograd.grad(dx, x_physics, torch.ones_like(dx), create_graph=True)[0]  # computes d^2y/dx^2
        #Todo: get back here and talk about pyhsiks!
        physics = dx2 + yhp  # computes the residual of the  SINE differential equation
        loss2 = torch.mean(physics ** 2)
        #Todo: lets talk about the (1e-4) * torch.mean(physics ** 2)
        #Todo us lamda on Loss
        # backpropagate joint loss
        #Todo whts up if there is no training loss
        loss = loss1 +  loss2  # add two loss terms together

        loss.backward()
        optimizer.step()

        # plot the result as training progresses
        if (i + 1) % 150 == 0:

            yh = model(x).detach()
            xp = x_physics.detach()

            plot_result(x.cpu(), y.cpu(), x_data.cpu(), y_data.cpu(), yh.cpu(), xp.cpu())

            file = "plots/pinn_%.8i.png" % (i + 1)
            plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
            files.append(file)
            print("Loss bei Iteration",i,"ist: ",loss)

            if (i + 1) % 6000 == 0:
                plt.show()
            else:
                plt.close("all")

    save_gif_PIL("pinn.gif", files, fps=20, loop=0)


if __name__ == '__main__':
    print("hello world")
    print("version of Torch:",torch.__version__)
    #torch.cuda.is_available()
    Sine()
    #pinn_sine.start

    print("is cuda availiabel?",torch.cuda.is_available())
