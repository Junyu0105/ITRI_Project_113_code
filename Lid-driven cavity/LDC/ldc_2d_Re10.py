# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import warnings
import matplotlib.pyplot as plt
import torch
import matplotlib as mpl
import numpy as np
import scipy.integrate


from sympy import Symbol, Eq, Abs

import modulus.sym
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_2d import Rectangle
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.domain.inferencer import PointwiseInferencer
from modulus.sym.key import Key
from modulus.sym.eq.pdes.navier_stokes import NavierStokes
from modulus.sym.utils.io import (
    csv_to_dict,
    ValidatorPlotter,
    InferencerPlotter,
)

# Problem Settings
height = 0.1
width = 0.1
rho = 1.0
u_driven = 1.0
Re=10
nu = u_driven * width / Re  
print(f"Re = {Re}")

@modulus.sym.main(config_path="conf", config_name="config_ldc")
def run(cfg: ModulusConfig) -> None:
    # make list of nodes to unroll graph on
    ns = NavierStokes(nu=nu, rho=rho, dim=2, time=False)
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = ns.make_nodes() + [flow_net.make_node(name="flow_network")]

    # add constraints to solver
    # make geometry
    x, y = Symbol("x"), Symbol("y")
    rec = Rectangle((-width / 2, -height / 2), (width / 2, height / 2))

    # make ldc domain
    ldc_domain = Domain()

    # top wall
    top_wall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"u": u_driven, "v": 0},
        batch_size=cfg.batch_size.TopWall,
        lambda_weighting={"u": 1.0 - 20 * Abs(x), "v": 1.0},  # weight edges to be zero
        criteria=Eq(y, height / 2),
    )
    ldc_domain.add_constraint(top_wall, "top_wall")

    # no slip
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"u": 0, "v": 0},
        batch_size=cfg.batch_size.NoSlip,
        criteria=y < height / 2,
    )
    ldc_domain.add_constraint(no_slip, "no_slip")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        batch_size=cfg.batch_size.Interior,
        lambda_weighting={
            "continuity": Symbol("sdf"),
            "momentum_x": Symbol("sdf"),
            "momentum_y": Symbol("sdf"),
        },
    )
    ldc_domain.add_constraint(interior, "interior")

    # add validator
    file_path = "openfoam/cavity_uniformVel0.csv"
    if os.path.exists(to_absolute_path(file_path)):
        mapping = {"Points:0": "x", "Points:1": "y", "U:0": "u", "U:1": "v", "p": "p"}
        openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping)
        openfoam_var["x"] += -width / 2  # center OpenFoam data
        openfoam_var["y"] += -height / 2  # center OpenFoam data
        openfoam_invar_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["x", "y"]
        }
        openfoam_outvar_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["u", "v"]
        }
        openfoam_validator = PointwiseValidator(
            nodes=nodes,
            invar=openfoam_invar_numpy,
            true_outvar=openfoam_outvar_numpy,
            batch_size=1024,
            plotter=ValidatorPlotter(),
        )
        ldc_domain.add_validator(openfoam_validator)

        # add inferencer data
        grid_inference = PointwiseInferencer(
            nodes=nodes,
            invar=openfoam_invar_numpy,
            output_names=["u", "v", "p"],
            batch_size=1024,
            plotter=InferencerPlotter(),
        )
        ldc_domain.add_inferencer(grid_inference, "inf_data")
    else:
        warnings.warn(
            f"Directory {file_path} does not exist. Will skip adding validators. Please download the additional files from NGC https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/resources/modulus_sym_examples_supplemental_materials"
        )

    # make solver
    slv = Solver(cfg, ldc_domain)

    # start solver
    slv.solve()

    plot_num = 101
    x = torch.linspace(-width / 2, width / 2,  plot_num)
    y = torch.linspace(-height / 2, height / 2, plot_num)
    X, Y = torch.meshgrid(x, y)

    # X_grid, Y_grid, u_plot_reshaped, v_plot_reshaped
    unique_sorted_x = np.unique(X.cpu().detach().numpy().flatten())
    unique_sorted_y = np.unique(Y.cpu().detach().numpy().flatten())
    X_grid, Y_grid = np.meshgrid(unique_sorted_x, unique_sorted_y)
    
    x_tensor = torch.tensor(X_grid.flatten(), dtype=torch.float32).cuda().reshape(-1, 1)
    y_tensor = torch.tensor(Y_grid.flatten(), dtype=torch.float32).cuda().reshape(-1, 1)
   
    inputs = {"x": x_tensor, "y": y_tensor}
    outputs = flow_net(inputs)
    u_predict = outputs["u"].cpu().detach().numpy().flatten()
    v_predict = outputs["v"].cpu().detach().numpy().flatten()
    p_predict = outputs["p"].cpu().detach().numpy().flatten()

    velocity = np.sqrt(u_predict**2 + v_predict**2)

    # Plotting the computed solution
    x_plot, y_plot = X.cpu().detach().numpy().flatten(), Y.cpu().detach().numpy().flatten()
    Plot2D(y_plot, x_plot, u_predict, name=f"PINN_u_Re{Re}", plot_type="contour")
    Plot2D(y_plot, x_plot, v_predict, name=f"PINN_v_Re{Re}", plot_type="contour")
    Plot2D(y_plot, x_plot, p_predict, name=f"PINN_p_Re{Re}", plot_type="contour")

    u_plot_reshaped = u_predict.reshape(X_grid.shape)
    v_plot_reshaped = v_predict.reshape(X_grid.shape)
    Plot2D(X_grid, Y_grid, u_plot_reshaped, v_plot_reshaped, name=f"Re{Re}_streamline", plot_type="streamline")

    Plot2D(y_plot, x_plot, velocity, name=f"Re{Re}_VelocityDistribution", flag=1, plot_type="contour")

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time: {training_time:.2f} seconds")
      
def Plot2D(X, Y, u_plot, v_plot=None, name="", flag=3, plot_type="contour"):
    print(f"Plot Solution {name} in 2D contour")
    plt.clf()
    fig, axes = plt.subplots(1, 1, figsize=(10, 8))

    if plot_type == "contour":
        if flag == 1:
            value_min, value_max = 0, 1.
        elif flag == 2:
            value_min, value_max = -0.4, 0.39
        else:
            value_min, value_max = np.min(u_plot), np.max(u_plot)
            # value_min, value_max = np.min(u_plot), np.max(u_plot)
        t1 = axes.tricontourf(X, Y, u_plot, cmap='jet', extend="both", levels=np.linspace(value_min, value_max, 101))
        axes.set_xlabel(r'$x$', fontsize=25)
        axes.set_ylabel(r'$y$', fontsize=25)
        axes.tick_params(axis='both', which='major', labelsize=20)
        axes.tick_params(axis='both', which='minor', labelsize=18)
        axes.set_title(f"Re = {Re}", fontsize=25)
        bar = fig.colorbar(t1)
        ticks = np.linspace(value_min, value_max, 5)
        bar.set_ticks(ticks)
        bar.set_ticklabels([f'{tick:.2f}' for tick in ticks])
        bar.ax.tick_params(labelsize=20)

    elif plot_type == "streamline" and v_plot is not None:
        strm = axes.streamplot(X, Y, u_plot, v_plot, 
                               color=np.sqrt(u_plot**2 + v_plot**2), 
                               cmap='jet', linewidth=2, density=2)
        bar = fig.colorbar(strm.lines)
        ticks = [0, 0.25, 0.5, 0.75, 1.0]
        bar.set_ticks(ticks)
        bar.set_label('velocity', fontsize=20)
        bar.ax.tick_params(labelsize=20)

        axes.set_xlim(-0.05, 0.05)
        axes.set_ylim(-0.05, 0.05)

        axes.set_xlabel(r'$x$', fontsize=25)
        axes.set_ylabel(r'$y$', fontsize=25)

        axes.tick_params(axis='both', which='major', labelsize=20)
        axes.tick_params(axis='both', which='minor', labelsize=20)

    plt.savefig(f'ldc_{name}.png', dpi=500, bbox_inches='tight', transparent=True)
    print(f"figure size = {plt.rcParams['figure.figsize']}")
    mpl.pyplot.close()
    return

if __name__ == "__main__":
    run()
    