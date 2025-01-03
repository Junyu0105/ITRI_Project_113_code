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
from sympy import Symbol, Eq, And, Or

import modulus.sym
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.utils.io import csv_to_dict
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_2d import Rectangle, Line, Channel2D
from modulus.sym.utils.sympy.functions import parabola
from modulus.sym.eq.pdes.navier_stokes import NavierStokes
from modulus.sym.eq.pdes.basic import NormalDotVec
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)

from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.key import Key
from modulus.sym.node import Node
from modulus.sym.models.fourier_net import FourierNetArch
import time
start_time = time.time()

# Problem settings
chip_height = 0.2
chip_width = 0.6/chip_height
chip_pos = -0.5 - chip_width/2
print(f"This case is for chip height = {chip_height}")

@modulus.sym.main(config_path="conf", config_name="config_flow")
def run(cfg: ModulusConfig) -> None:
    #############
    # Real Params
    #############
    fluid_kinematic_viscosity = 0.004195088  # m**2/s
    fluid_density = 1.1614  # kg/m**3
    fluid_specific_heat = 1005  # J/(kg K)
    fluid_conductivity = 0.0261  # W/(m K)

    # copper params
    copper_density = 8930  # kg/m3
    copper_specific_heat = 385  # J/(kg K)
    copper_conductivity = 385  # W/(m K)

    # boundary params
    inlet_velocity = 5.24386  # m/s
    inlet_temp = 25.0  # C
    copper_heat_flux = 51.948051948  # W / m2

    ################
    # Non dim params
    ################
    length_scale = 0.04  # m
    time_scale = 0.007627968710072352  # s
    mass_scale = 7.43296e-05  # kg
    temp_scale = 1.0  # K
    velocity_scale = length_scale / time_scale  # m/s
    pressure_scale = mass_scale / (length_scale * time_scale**2)  # kg / (m s**2)
    density_scale = mass_scale / length_scale**3  # kg/m3
    watt_scale = (mass_scale * length_scale**2) / (time_scale**3)  # kg m**2 / s**3
    joule_scale = (mass_scale * length_scale**2) / (
        time_scale**2
    )  # kg * m**2 / s**2

    ##############################
    # Nondimensionalization Params
    ##############################
    # fluid params
    nd_fluid_kinematic_viscosity = fluid_kinematic_viscosity / (
        length_scale**2 / time_scale
    )
    nd_fluid_density = fluid_density / density_scale
    nd_fluid_specific_heat = fluid_specific_heat / (
        joule_scale / (mass_scale * temp_scale)
    )
    nd_fluid_conductivity = fluid_conductivity / (
        watt_scale / (length_scale * temp_scale)
    )
    nd_fluid_diffusivity = nd_fluid_conductivity / (
        nd_fluid_specific_heat * nd_fluid_density
    )

    # copper params
    nd_copper_density = copper_density / (mass_scale / length_scale**3)
    nd_copper_specific_heat = copper_specific_heat / (
        joule_scale / (mass_scale * temp_scale)
    )
    nd_copper_conductivity = copper_conductivity / (
        watt_scale / (length_scale * temp_scale)
    )
    nd_copper_diffusivity = nd_copper_conductivity / (
        nd_copper_specific_heat * nd_copper_density
    )

    # boundary params
    nd_inlet_velocity = inlet_velocity / (length_scale / time_scale)
    nd_inlet_temp = inlet_temp / temp_scale
    nd_copper_source_grad = copper_heat_flux * length_scale / temp_scale

    # make list of nodes to unroll graph on
    ns = NavierStokes(
        nu=nd_fluid_kinematic_viscosity, rho=nd_fluid_density, dim=2, time=False
    )
    normal_dot_vel = NormalDotVec(["u", "v"])
    flow_net = FourierNetArch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        frequencies=("axis", [i / 5.0 for i in range(25)]),
    )
    nodes = (
        ns.make_nodes()
        + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network")]
    )

    # add constraints to solver
    # simulation params
    channel_length = (-2.5, 5.0)
    channel_width = (-0.5, 0.5)

    # define sympy variables to parametrize domain curves
    x, y = Symbol("x"), Symbol("y")

    # define geometry
    channel = Channel2D(
        (channel_length[0], channel_width[0]), (channel_length[1], channel_width[1])
    )
    inlet = Line(
        (channel_length[0], channel_width[0]),
        (channel_length[0], channel_width[1]),
        normal=1,
    )
    outlet = Line(
        (channel_length[1], channel_width[0]),
        (channel_length[1], channel_width[1]),
        normal=1,
    )
    rec = Rectangle(
        (chip_pos, channel_width[0]),
        (chip_pos + chip_width, channel_width[0] + chip_height),
    )
    geo = channel - rec
    x_pos = Symbol("x_pos")
    integral_line = Line((x_pos, channel_width[0]), (x_pos, channel_width[1]), 1)
    x_pos_range = {
        x_pos: lambda batch_size: np.full(
            (batch_size, 1), np.random.uniform(channel_length[0], channel_length[1])
        )
    }

    # make domain
    domain = Domain()

    # inlet
    inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet,
        outvar={"u": nd_inlet_velocity, "v": 0},
        batch_size=cfg.batch_size.inlet,
    )
    domain.add_constraint(inlet, "inlet")

    # outlet
    outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet,
        outvar={"p": 0},
        batch_size=cfg.batch_size.outlet,
        criteria=Eq(x, channel_length[1]),
    )
    domain.add_constraint(outlet, "outlet")

    # no slip
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 0, "v": 0},
        batch_size=cfg.batch_size.no_slip,
    )
    domain.add_constraint(no_slip, "no_slip")

    # interior lr
    interior_lr = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        batch_size=cfg.batch_size.interior_lr,
        criteria=Or(x < (chip_pos - 0.25), x > (chip_pos + chip_width + 0.25)),
        lambda_weighting={
            "continuity": 2 * Symbol("sdf"),
            "momentum_x": 2 * Symbol("sdf"),
            "momentum_y": 2 * Symbol("sdf"),
        },
    )
    domain.add_constraint(interior_lr, "interior_lr")

    # interior hr
    interior_hr = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        batch_size=cfg.batch_size.interior_hr,
        criteria=And(x > (chip_pos - 0.25), x < (chip_pos + chip_width + 0.25)),
        lambda_weighting={
            "continuity": 2 * Symbol("sdf"),
            "momentum_x": 2 * Symbol("sdf"),
            "momentum_y": 2 * Symbol("sdf"),
        },
    )
    domain.add_constraint(interior_hr, "interior_hr")

    # integral continuity
    def integral_criteria(invar, params):
        sdf = geo.sdf(invar, params)
        return np.greater(sdf["sdf"], 0)

    integral_continuity = IntegralBoundaryConstraint(
        nodes=nodes,
        geometry=integral_line,
        outvar={"normal_dot_vel": 1},
        batch_size=cfg.batch_size.num_integral_continuity,
        integral_batch_size=cfg.batch_size.integral_continuity,
        lambda_weighting={"normal_dot_vel": 1},
        criteria=integral_criteria,
        parameterization=x_pos_range,
    )
    domain.add_constraint(integral_continuity, "integral_continuity")

    # add validation data
    file_path = "openfoam/2d_real_cht_fluid.csv"
    if os.path.exists(to_absolute_path(file_path)):
        mapping = {
            "x-coordinate": "x",
            "y-coordinate": "y",
            "x-velocity": "u",
            "y-velocity": "v",
            "pressure": "p",
            "temperature": "theta_f",
        }
        openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping)
        openfoam_var["x"] = openfoam_var["x"] / length_scale - 2.5  # normalize pos
        openfoam_var["y"] = openfoam_var["y"] / length_scale - 0.5
        openfoam_var["p"] = (openfoam_var["p"] + 400.0) / pressure_scale
        openfoam_var["u"] = openfoam_var["u"] / velocity_scale
        openfoam_var["v"] = openfoam_var["v"] / velocity_scale

        openfoam_invar_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["x", "y"]
        }
        openfoam_outvar_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["u", "v", "p"]
        }
        openfoam_validator = PointwiseValidator(
            nodes=nodes,
            invar=openfoam_invar_numpy,
            true_outvar=openfoam_outvar_numpy,
        )
        domain.add_validator(openfoam_validator)
    else:
        warnings.warn(
            f"Directory {file_path} does not exist. Will skip adding validators. Please download the additional files from NGC https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/resources/modulus_sym_examples_supplemental_materials"
        )

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()

    Post_processing(flow_net)

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time: {training_time:.2f} seconds")


def Post_processing(flow_net):
    plot_num = 201

    # Get fluid coordinate
    x = torch.linspace(-2.5, chip_pos, plot_num)
    y = torch.linspace(-0.5, 0.5, plot_num)
    X1, Y1 = torch.meshgrid(x, y)
    x_fluid_l, y_fluid_l = X1.reshape(-1, 1).cuda(), Y1.reshape(-1, 1).cuda()
    inputs = {"x": x_fluid_l, "y": y_fluid_l}
    outputs_l = flow_net(inputs)
    u_fluid_l = outputs_l["u"]
    v_fluid_l = outputs_l["v"]
    plot_l = (x_fluid_l, y_fluid_l, (u_fluid_l**2 + v_fluid_l**2)**0.5)

    x = torch.linspace(chip_pos, chip_pos + chip_width, plot_num)
    y = torch.linspace(-0.5 + chip_height, 0.5, plot_num)
    X2, Y2 = torch.meshgrid(x, y)
    x_fluid_m, y_fluid_m = X2.reshape(-1, 1).cuda(), Y2.reshape(-1, 1).cuda()
    inputs = {"x": x_fluid_m, "y": y_fluid_m}
    outputs_m = flow_net(inputs)
    u_fluid_m = outputs_m["u"]
    v_fluid_m = outputs_m["v"]
    plot_m = (x_fluid_m, y_fluid_m, (u_fluid_m**2 + v_fluid_m**2)**0.5)

    x = torch.linspace(chip_pos + chip_width,   5, plot_num)
    y = torch.linspace(-0.5, 0.5, plot_num)
    X3, Y3 = torch.meshgrid(x, y)
    x_fluid_r, y_fluid_r = X3.reshape(-1, 1).cuda(), Y3.reshape(-1, 1).cuda()
    inputs = {"x": x_fluid_r, "y": y_fluid_r}
    outputs_r = flow_net(inputs)
    u_fluid_r = outputs_r["u"]
    v_fluid_r = outputs_r["v"]
    plot_r = (x_fluid_r, y_fluid_r, (u_fluid_r**2 + v_fluid_r**2)**0.5)

    u_l = u_fluid_l.cpu().detach().numpy()
    u_m = u_fluid_m.cpu().detach().numpy()
    u_r = u_fluid_r.cpu().detach().numpy()
    v_l = v_fluid_l.cpu().detach().numpy()
    v_m = v_fluid_m.cpu().detach().numpy()
    v_r = v_fluid_r.cpu().detach().numpy()

    u_all = np.concatenate([u_l, u_m, u_r])
    v_all = np.concatenate([v_l, v_m, v_r])
    np.savetxt(f"Height{chip_height}_u.txt", u_all)
    np.savetxt(f"Height{chip_height}_v.txt", v_all)

    Plot_fluid(plot_l, plot_m, plot_r, f"fluidV_h{chip_height}", 1)

def Plot_fluid(plot_l, plot_m, plot_r, name, flag=3):
    plot_l = [tensor.cpu().detach().numpy().flatten() for tensor in plot_l]
    plot_m = [tensor.cpu().detach().numpy().flatten() for tensor in plot_m]
    plot_r = [tensor.cpu().detach().numpy().flatten() for tensor in plot_r]
    X_l, Y_l, velocity_l = plot_l
    X_m, Y_m, velocity_m = plot_m
    X_r, Y_r, velocity_r = plot_r
    x_all = np.concatenate([X_l, X_m, X_r])
    y_all = np.concatenate([Y_l, Y_m, Y_r])
    np.savetxt(f"Height{chip_height}_x.txt", x_all)
    np.savetxt(f"Height{chip_height}_y.txt", y_all)
    print(f"Plot prediction {name} in 2D contour")
    plt.clf()
    fig, axes = plt.subplots(1,1,figsize=(8,1))
    if flag == 1:
        value_min, value_max = 0, 7.3
    else:
        value_min, value_max = np.min(velocity_r), np.max(velocity_m)
    t1 = axes.tricontourf(X_l, Y_l, velocity_l, cmap='jet', levels=np.linspace(value_min, value_max,101))
    t2 = axes.tricontourf(X_m, Y_m, velocity_m, cmap='jet', levels=np.linspace(value_min, value_max,101))
    t3 = axes.tricontourf(X_r, Y_r, velocity_r, cmap='jet', levels=np.linspace(value_min, value_max,101))
    # axes.set_xlabel(r'$x$', fontsize=20)
    # axes.set_ylabel(r'$y$', fontsize=20)
    # bar = fig.colorbar(t1)
    # ticks = np.linspace(value_min, value_max, 5)
    # bar.set_ticks(ticks)
    # bar.set_ticklabels([f'{tick:.1f}' for tick in ticks])
    plt.savefig(f'{name}.png', dpi=500, bbox_inches='tight', transparent=True)
    mpl.pyplot.close()
    return

if __name__ == "__main__":
    run()
