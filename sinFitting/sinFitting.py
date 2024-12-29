import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib as mpl
from sympy import Symbol, sin, pi, Function

import modulus.sym
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_1d import Line1D
from modulus.sym.domain.constraint import (
    PointwiseInteriorConstraint,
)

from modulus.sym.key import Key
from modulus.sym.eq.pde import PDE

class Fitting(PDE):

    name = "Fitting"

    def __init__(self):
        # coordinates
        x = Symbol("x")

        # make input variables by a dictionary
        input_variables = {"x": x}

        # make a function named "u" which the input is x and y (input_variable)
        y = Function("y")(*input_variables)

        # set equations by a dictionary
        self.equations = {}
        self.equations["prediction"] = y

# It's a decorator to set and initialize the environment of Modulus library
@modulus.sym.main(config_path="conf", config_name="config_1Dwave")

# Main function named "run" whose input is config file including batching size and some training parameters
def run(cfg: ModulusConfig) -> None:
    # make list of nodes to unroll graph on
    # Get the class and input a constant variable of wave speed coefficient
    sf = Fitting()
    # Set a architecture of NN whose input is x and t, ouput is u and the other settings can refer the cfg file
    fitting_net = instantiate_arch(
        input_keys=[Key("x")],
        output_keys=[Key("y")],
        cfg=cfg.arch.fully_connected,
    )
    # sf.make_nodes() transform the function into nodes in the graph
    # [fitting_net.make_node(name="Fitting_network")] transform the NN architecture to nodes in the graph
    nodes = sf.make_nodes() + [fitting_net.make_node(name="Fitting_network")]

    # add constraints to solver
    # make geometry (problem settings)
    x = Symbol("x")
    L = 1
    geo = Line1D(0, L)

    # make domain
    domain = Domain()

    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"prediction": x + sin(4*pi*x)},
        batch_size=cfg.batch_size.interior,
        # parameterization=time_range,
    )
    domain.add_constraint(interior, "interior")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()

    # For illustration, assuming a placeholder for u computed from validation or solver
    x_num = 100
    x = torch.linspace(0, L, x_num).reshape(-1, 1).cuda()
    # Create input dictionary for the model
    inputs = {"x": x}
    # Pass inputs through the model to get predictions
    outputs = fitting_net(inputs)
    y_predict = outputs["y"].cpu().detach().numpy().flatten()
    y_data = (x + torch.sin(4*torch.pi*x)).cpu().detach().numpy().flatten()

    # Plotting the computed solution
    x_plot = x.cpu().detach().numpy().flatten()
    plt.plot(x_plot, y_data, label='data', color='blue')
    plt.plot(x_plot, y_predict, label='NN', color="red", linestyle='--')
    plt.legend()
    plt.title('Fitting Test')
    plt.savefig(f'Fitting_test.png', dpi=500, bbox_inches='tight', transparent=True)
    print(f"figure size = {plt.rcParams['figure.figsize']}")
    mpl.pyplot.close()


if __name__ == "__main__":
    run()
