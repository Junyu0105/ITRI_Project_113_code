import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib as mpl
from sympy import Symbol, sin, Function, Number

import modulus.sym
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_1d import Line1D
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)

from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.key import Key
from modulus.sym.node import Node
from modulus.sym.eq.pde import PDE

class WaveEquation1D(PDE):
    """
    Wave equation 1D
    The equation is given as an example for implementing
    your own PDE. A more universal implementation of the
    wave equation can be found by
    `from modulus.sym.eq.pdes.wave_equation import WaveEquation`.

    Parameters
    ==========
    c : float, string
        Wave speed coefficient. If a string then the
        wave speed is input into the equation.
    """

    name = "WaveEquation1D"

    def __init__(self, c=1.0):
        # coordinates
        x = Symbol("x")

        # time
        t = Symbol("t")

        # make input variables by a dictionary
        input_variables = {"x": x, "t": t}

        # make a function named "u" which the input is x and y (input_variable)
        u = Function("u")(*input_variables)

        # wave speed coefficient
        if type(c) is str:
            # If it is a string, c is dependent to x and t. Therefore, view it like a function named "c"
            c = Function(c)(*input_variables)
        elif type(c) in [float, int]:
            # If it is a float or int, c is a coefficient
            c = Number(c)

        # set equations by a dictionary
        self.equations = {}
        self.equations["wave_equation"] = u.diff(t, 2) - (c**2 * u.diff(x)).diff(x)

# It's a decorator to set and initialize the environment of Modulus library
@modulus.sym.main(config_path="conf", config_name="config_1Dwave")

# Main function named "run" whose input is config file including batching size and some training parameters
def run(cfg: ModulusConfig) -> None:
    # make list of nodes to unroll graph on
    # Get the class and input a constant variable of wave speed coefficient
    we = WaveEquation1D(c=1.0)
    # Set a architecture of NN whose input is x and t, ouput is u and the other settings can refer the cfg file
    wave_net = instantiate_arch(
        input_keys=[Key("x"), Key("t")],
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )
    # we.make_nodes() transform the function into nodes in the graph
    # [wave_net.make_node(name="wave_network")] transform the NN architecture to nodes in the graph
    nodes = we.make_nodes() + [wave_net.make_node(name="wave_network")]

    # add constraints to solver
    # make geometry (problem settings)
    x, t_symbol = Symbol("x"), Symbol("t")
    L = float(np.pi)
    geo = Line1D(0, L)
    time_range = {t_symbol: (0, 2 * L)}

    # make domain
    domain = Domain()

    # initial condition
    IC = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": sin(x), "u__t": sin(x)}, # Initial condition
        batch_size=cfg.batch_size.IC,
        lambda_weighting={"u": 1.0, "u__t": 1.0}, # The penalty weight
        parameterization={t_symbol: 0.0}, # Independent to time variable
    )
    domain.add_constraint(IC, "IC")

    # boundary condition
    BC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 0},
        batch_size=cfg.batch_size.BC,
        parameterization=time_range,
    )
    domain.add_constraint(BC, "BC")

    # interior (governing equation)
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"wave_equation": 0},
        batch_size=cfg.batch_size.interior,
        parameterization=time_range,
    )
    domain.add_constraint(interior, "interior")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()

    # For illustration, assuming a placeholder for u computed from validation or solver
    deltaT = 0.1
    deltaX = 0.1
    x = torch.arange(0, L, deltaX)
    t = torch.arange(0, 2 * L, deltaT)
    X, T = torch.meshgrid(x, t)
    # Flatten the grid arrays to  create inputs for themodel
    x_tensor, t_tensor = X.reshape(-1, 1).cuda(), T.reshape(-1, 1).cuda()
    # Create input dictionary for the model
    inputs = {"x": x_tensor, "t": t_tensor}
    # Pass inputs through the model to get predictions
    outputs = wave_net(inputs)
    u_predict = outputs["u"].cpu().detach().numpy().flatten()
    u_exact = (torch.sin(X) * (torch.cos(T) + torch.sin(T))).cpu().detach().numpy().flatten()

    # Plotting the computed solution
    x_plot, t_plot = X.cpu().detach().numpy().flatten(), T.cpu().detach().numpy().flatten()
    Plot2D(x_plot, t_plot, u_exact, "exact")
    Plot2D(x_plot, t_plot, u_predict, "PINN")
    Plot2D(x_plot, t_plot, u_predict - u_exact, "Error", True)


def Plot2D(X, T, u_plot, name, flag=False):
    print(f"Plot Solution in 2D contour")
    plt.clf()
    fig, axes = plt.subplots(1,1,figsize=(4,6))
    if flag:
        value_min, value_max = np.min(u_plot), np.max(u_plot)
        value_min, value_max = -0.006, 0.006
    else:
        value_min, value_max = -1.5, 1.5
    t1 = axes.tricontourf(X, T, u_plot, cmap='jet', extend="both", levels=np.linspace(value_min, value_max,101))
    axes.set_xlabel(r'$x$')
    axes.set_ylabel(r'$t$')
    bar = fig.colorbar(t1)
    ticks = np.linspace(value_min, value_max,5)
    bar.set_ticks(ticks)
    if flag:
        tick_labels = [f"{tick:.3f}" for tick in ticks]
    else:
        tick_labels = [f"{tick:.2f}" for tick in ticks]
    bar.set_ticklabels(tick_labels)
    plt.savefig(f'1Dwave_{name}.png', dpi=500, bbox_inches='tight', transparent=True)
    print(f"figure size = {plt.rcParams['figure.figsize']}")
    mpl.pyplot.close()
    return

if __name__ == "__main__":
    run()
