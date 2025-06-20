from dataclasses import dataclass
import datetime as dt
from enum import Enum
import json
import math
import numpy as np
from numpy.typing import NDArray
import pyomo.environ as pyo
import pandas as pd
from pathlib import Path
import scipy.sparse as sp
from typing import Dict, List, Optional, NewType
import wntr

# assumptions:
# - WWMD resolution is apppopriate (this is more granular than the DMA)
# - if the time horizon of sensor data is longer than the model, we slice the
#   sensor data so they align. instead, we could explore taking an average, 
#   statistical analysis of the distribution of sensor data, etc. We should also 
#   handle if sensor data time horizon is less than the model.

network_path = Path("data/network/BWFLnet_2023_04.inp")
model = wntr.network.WaterNetworkModel(network_path)
simulator = wntr.sim.EpanetSimulator(model)
simulation_results = simulator.run_sim()

# TODO: plot data to better understand demand scaling
#     - i'm not sure exacly _what_ to plot yet to validate demand scaling.
#     - map of the WWMDs, their demands, model and sensor demands of the 
#      `flow_in` and `flow_out` nodes may be compelling.

# TODO: replace with live sensor data. for now i'll either use Bradley's data
# (as i am now) or stochastic data (this may lend itself to writing tests).

# TODO: in writing tests i realized it's kind of awkward to have a list of 
# `SensorReading` rather than a `DataFrame`. it also would be useful to better
# protect against model and sensor data have different time cadences.

# TODO: consider calibration coefficients from Bradley's implementation.

# TODO: more descriptive names for optimization parameters.

# TODO: install MyPy for type checking?

# TODO: make sure i'm only using DataFrames/Series when it's actually useful.

# TODO: why does Bradley use `valve_info.json` rather than getting valve info
# from the model, do they differ?

@dataclass
class SensorReading:
    datetime: dt.datetime
    bwfl_id: str
    dma_id: str
    wwmd_id: str
    min: Optional[float]
    mean: Optional[float]
    max: Optional[float]

    def __post_init__(self):
        self.min = self.min if pd.notnull(self.min) else None
        self.mean = self.mean if pd.notnull(self.mean) else None
        self.max = self.max if pd.notnull(self.max) else None

@dataclass
class OptimizationParameters:
    # reservoir index x datetime
    h0: pd.DataFrame
    # datetime x junction
    d: pd.DataFrame

# TODO: switch to `NDArray[int]` as its more type precise?
@dataclass
class IncidentMatrices:
    # link x junction
    A12: sp.csr_matrix
    # link x source
    A10: sp.csr_matrix

    @classmethod
    def from_model(cls, model: wntr.network.WaterNetworkModel):
        link_junction = np.zeros(
            (model.num_links, model.num_junctions), 
            dtype = int
        )
        link_reservoir = np.zeros(
            (model.num_links, model.num_reservoirs), 
            dtype = int
        )
        # start node is implies -1 as flow is leaving, end node implies 1 as
        # flow is entering (flow is from the link's start to end node;
        # confirmed via EPANET).
        for i, (_, link) in enumerate(model.links()):
            for j, (junction_name, _) in enumerate(model.junctions()):
                if link.start_node_name == junction_name:
                    link_junction[i, j] = -1
                elif link.end_node_name == junction_name:
                    link_junction[i, j] = 1
            for k, (reservoir_name, _) in enumerate(model.reservoirs()):
                # TODO: is entering a source actually valid? according to our
                # slides it is...
                if link.start_node_name == reservoir_name:
                    link_reservoir[i, k] = -1
                elif link.end_node_name == reservoir_name:
                    link_reservoir[i, k] = 1
        # store as Compressed Sparse Row matrices to save space.
        return cls(
            A12 = sp.csr_matrix(link_junction),
            A10 = sp.csr_matrix(link_reservoir)
        )

@dataclass
class HeadlossCoefficients:
    # Bradley calls this `K` but i'm sticking to the notation in the slides
    R: float
    n_exp: float

    # TODO: use string enum
    # i assume the model is calibrated and use the `roughness` and `minor_loss` 
    # properties stored on the model. If not, i may need to read in calibration
    # coefficients from a seperate data source.
    @classmethod
    def from_link(
        cls, 
        link: wntr.network.Link,
        model: wntr.network.WaterNetworkModel
    ):
        R = 0
        n_exp = 0
        assert_msg = f"Link's link_type property doesn't match its type: "
        "link=\{link} link_type=\{link.link_type}"
        match link:
            case wntr.network.Pipe() as pipe:
                assert(pipe.link_type == "pipe", assert_msg)
                match model.options.hydraulic.headloss:
                    case 'H-W':
                        n_exp = 1.852
                        R = (
                            10.67 * pipe.length
                        ) / (
                            (
                                pipe.roughness ** n_exp
                            ) * (
                                pipe.diameter ** 4.8704
                            )
                        )
                    case 'D-W', 'C-M':
                        raise Exception(
                            f"Support not yet available for headloss formula: "
                            "\{model.options.hydraulic.headloss}"
                        )
                    case _:
                        raise Exception(
                            f"Unknown headloss formula: "
                            "\{model.options.hydraulic.headloss}"
                        )
            case wntr.network.Valve() as valve:
                assert(valve.link_type == "valve", assert_msg)
                n_exp = 2
                # `Valve.minor_loss` isn't documented, but its used in the 
                # `wntr` source for this purpose, so i'm leveraging it here.
                R = 8.0 * valve.minor_loss / (
                    9.81 * (math.pi ** 2) * (valve.diameter) ** 4
                )
            case wntr.network.Pump() as pump:
                assert(pump.link_type == "pump", assert_msg)
                raise Exception(
                    f"Support not yet available for \{pump.link_type}"
                )
            case _:
                raise Exception(f"Unexpected link_type: \{link.link_type}")
        return cls(R = R, n_exp = n_exp)

WWMD_ID = NewType('WWMD_ID', str)
BWFL_ID = NewType('BWFL_ID', str)
# TODO: update Python to 3.11+ and use StrEnum.
# class FlowMapKey(StrEnum):
#     FLOW_IN = 'flow_in'
#     FLOW_OUT = 'flow_out'
# TODO: update Python to 3.11+ and use StrEnum.
FlowMap = Dict[WWMD_ID, Dict[str, List[BWFL_ID]]]

# TODO: convert from flow to volume
def scale_demands(
    model_demand_table: pd.DataFrame,
    flow_map: FlowMap, 
    flow_readings: List[SensorReading],
    epanet_id_lookup_table: pd.DataFrame
) -> pd.DataFrame:
    
    """Scales modeled demand for each modelled timestamp according to the 
    relative demand of each WWMD according to live sensors. Sensor data is 
    sliced match the time horizion of model data.

    :param model_demand_table: A table describing the modelled demand for each
    EPANET node. Indexed by timesteps. Start time, timestep cadence, and flow 
    units shoud align with `flow_readings`.
    :param flow_map: A map of in/outflow `BWFL_IDs` for WWMDs. Likely decoded
    from JSON.
    :param flow_readings: A list of field sensor readings to use to scale model
    demand. Start time, timestep cadence, and flow units shoud align with 
    `model_demand_table`.
    :param epanet_id_lookup_table: A table describing the `WWMD_ID`s for each
    EPANET node ID. Note there tend to multiple EPANET nodes in a given WWMD.

    :returns: A data frame describing the scaled demands of the nodes (columns)
    for the modelled timestamps.
    """

    scaled_demands = pd.DataFrame(
        data = None,
        index = model_demand_table.index
    )

    for wwmd_id, flows in flow_map.items():
        # find the EPANET nodes in the current WWMD using the lookup table. 
        # filter the model demands to just those nodes.
        epanet_node_ids = epanet_id_lookup_table.loc[
            epanet_id_lookup_table["WWMD ID"] == wwmd_id
        ]["EPANET Node ID"]
        if epanet_node_ids.empty:
            raise ValueError(
                f"No EPANET nodes were mapped to the current wwmd_id. "
                f"wwmd_id={wwmd_id}"
            )
        
        # TODO: is there a better way to scale this than just using the mean?

        # calculate the total in/outflow for each WWMD (for each timestamp) 
        # using sensor data.
        # TODO: update Python to 3.11+ and use StrEnum.
        flow_in = pd.DataFrame(
            data = [
                reading.__dict__
                for reading in flow_readings
                if reading.bwfl_id in flows['flow_in']
            ],
            columns = ["datetime", "mean"]
        ).groupby('datetime')["mean"].sum()

        # TODO: update Python to 3.11+ and use StrEnum.
        flow_out = pd.DataFrame(
            data = [
                reading.__dict__
                for reading in flow_readings 
                if reading.bwfl_id in flows['flow_out']
            ],
            columns = ["datetime", "mean"]
        ).groupby('datetime')["mean"].sum()

        # the demand of the current WWMD according to field data.
        sensor_demands_wwmd = flow_in.sub(flow_out, fill_value = 0)
        # the demand of the current WWMD according to model data.
        model_demand_table_wwmd = model_demand_table[epanet_node_ids]
        # scaling factor is division of WWMD demand by the total modelled 
        # demand.
        # note we only take the slice of `sensor_demands` that correspond to
        # `model_demands` as the time horizons may be different. but we assume
        # timesteps are at the same cadence and start times align.
        # TODO: handle division by 0
        scaling_factors = sensor_demands_wwmd.array[
            :len(model_demand_table_wwmd)
        ] / model_demand_table_wwmd.sum(axis = 'columns')

        scaled_demands = scaled_demands.join(
            model_demand_table_wwmd.mul(scaling_factors, axis = 'index')
        )
        # sort nodes lexicographically (for easier debugging). 
        # this is a specific command as it can't be achieved via the index-based
        # `join(...)` above.
        scaled_demands.sort_index(axis = 'columns', inplace = True)

    # note some of the modelled demands are 0 (e.g., `node_0004`)
    return scaled_demands

def load_sensor_readings(path: Path) -> List[SensorReading]:
        # produces a list of dictionaries mapping property/column name to value 
        # for each instance.
        dictionaries = pd.read_csv(
            path,
            parse_dates = True
        ).to_dict(orient = "records")
        return [SensorReading(**dictionary) for dictionary in dictionaries]

def create_h0(
    pressure_readings: List[SensorReading],
    epanet_id_lookup_table: pd.DataFrame
) -> pd.DataFrame:
    # some of these devices aren't in the WMWDs i'm looking at. as an 
    # optimization i should probably remove those and any columnds in don't 
    # need.
    # does it make sense to do this for all sensor readings over the entire
    # time horizon, or just the ones we use to scale?
    # lookup table?
    pressure_device_lookup_table_path = Path(
        "data/devices/pressure_device_database.xlsx"
    )
    pressure_device_lookup_table = pd.read_excel(
        pressure_device_lookup_table_path,
        dtype = { 'Asset ID': str }
    )

    h0 = pd.DataFrame(
        None,
        index = range(model.num_reservoirs),
        columns = list(set(reading.datetime for reading in pressure_readings))
    )
    for i, reservoir_model_name in enumerate(model.reservoir_name_list):
        # is there a way to condense this into one line?
        infoworks_node_id = epanet_id_lookup_table.loc[
            epanet_id_lookup_table["EPANET Node ID"] == reservoir_model_name
        ]["InfoWorks Node ID"].item()

        # TODO: below i just grab first match as bradley does in 
        # optimal-prv-control.py:118.
        # looking at the data, i'm concerned this may cause us to select 
        # inactive devices that don't correspond to the sensor reading time
        # period.
        
        # `infoworks_node_id` is equivalent to Asset ID in the device table.
        pressure_device_lookup_table_row = pressure_device_lookup_table.loc[
            pressure_device_lookup_table["Asset ID"] == infoworks_node_id
        ].to_numpy()[0]
        pressure_device_lookup_table_row = pd.Series(
            pressure_device_lookup_table_row,
            index = pressure_device_lookup_table.columns
        )

        elevation_head = pressure_device_lookup_table_row["Final Elevation (m)"]

        # `infoworks_node_id` is equivalent to Asset ID in the device table.
        bwfl_id = pressure_device_lookup_table_row["BWFL ID"]

        # TODO: use series
        pressure_head = pd.DataFrame(
            data = [
                reading.__dict__
                for reading in pressure_readings
                if reading.bwfl_id == bwfl_id
            ],
            columns = ["datetime", "mean"]
        ).set_index("datetime")

        # TODO: check this
        head = pressure_head + elevation_head
        # TODO: is there a better way i can align the columns here since they 
        # are equivalent timestamps?
        h0.loc[i] = head["mean"]
    
    return h0

# elevations for each junction
def create_z(model: wntr.network.WaterNetworkModel) -> pd.Series:
    return pd.Series(
        [junction.elevation for (_, junction) in model.junctions()]
    )


# TODO: better type for `nt` a la index typealias?
# todo: rename `model`` to `network_model`?
def optimize(
    model: wntr.network.WaterNetworkModel, 
    nt: int, 
    z: pd.Series, 
    d: pd.DataFrame,
    h0: pd.DataFrame,
    A12: sp.csr_matrix,
    A10: sp.csr_matrix,
    link_headloss_coefficients: NDArray[np.object_]
):
    
    # we consider the max. value of all of the reservoirs over all the 
    # timestamps the upper limit on junction head.
    # TODO: wouldn't it be more precise to calculate the max per timestamp?
    max_reservoir_head = np.max(h0)
    def head_decision_variable_bounds(_, i, t):
        elevation = z[i]
        minimum_service_pressure = 0
        # TODO: probably reshape demand df so its same orientation as other
        # structs.
        # TODO: should there be a threshold?
        if d.iloc[t, i] <= 0:
            # if there's no demand at the junction, we only need to maintain a
            # 5m pressure head to avoid degradation of water quality.
            minimum_service_pressure = 5
        else:
            # if there's is demand at the junction, we maintain a 15m pressure 
            # head to meet demand.
            minimum_service_pressure = 15
        return (elevation + minimum_service_pressure, max_reservoir_head)

    print(f"valve count: \{model.num_valves}")
    valve_indices = [
            j
            for j, (_, link) in enumerate(model.links()) 
            # TODO: this check should match the assert.
            if link.link_type == "Valve"
        ]
    print(valve_indices)
    def valve_outlet_pressure_decision_variable_bounds(_, n, t):
        # for each valve, max. eta is difference of the start node max. head 
        # (outflow) and end node min. head (inflow)
        j = valve_indices[n]
        # extract first element as `nonzero` returns a tuple of arrays 
        # corresponding to each axis, even though our input is only 1D.
        out_index = np.nonzero(A12[j, :] == -1)[0].item()
        in_index = np.nonzero(A12[j, :] == 1)[0].item()
        out_head_max =  head_decision_variable_bounds(_, out_index, t)[1]
        in_head_min =  head_decision_variable_bounds(_, in_index, t)[0]
        # TODO: shouldn't we apply the same logic to eta min?
        return (0, out_head_max - in_head_min)

    optimization_model = pyo.ConcreteModel()
    # 0-indexed, per Bradley's implementation
    # it also could be cool to
    # to create an extension that indexes 0-index arrays with 1-index sets so the
    # math is clearer here.
    optimization_model.I = pyo.RangeSet(0, model.num_junctions - 1)
    optimization_model.J = pyo.RangeSet(0, model.num_links - 1)
    optimization_model.S = pyo.RangeSet(0, model.num_reservoirs - 1)
    optimization_model.N = pyo.RangeSet(0, model.num_valves - 1)
    optimization_model.T = pyo.RangeSet(0, nt - 1)
    
    optimization_model.h = pyo.Var(
        optimization_model.I,
        optimization_model.T,
        bounds = head_decision_variable_bounds,
        initialize = lambda _, i, t: h_0.iloc[t, i]
    )
    optimization_model.q = pyo.Var(
        optimization_model.J,
        optimization_model.T,
        bounds = (-100, 100),
        initialize = lambda _, j, t: q_0.iloc[t, j]
    )
    optimization_model.eta = pyo.Var(
        optimization_model.N,
        optimization_model.T,
        bounds = valve_outlet_pressure_decision_variable_bounds,
        initialize = 0
    )

    # TODO: add in epsilon.
    def energy_conservation_constraint(model, j, t):
        coefs = link_headloss_coefficients[j]
        return (
            coefs.R * model.q[j, t] * (abs(model.q[j, t]) ** (coefs.n_exp - 1))
            + sum(A12[j, i] * model.h[i, t] for i in model.I)
            + sum(A10[j, s] * h0.iloc[s, t] for s in model.S)
            + sum(model.eta[n, t] for n in model.N if j in valve_indices)
            == 0
        )
    
    def mass_conservation_constraint(model, j, t):
        return sum(A12[j, i] * model.q[j, t] for j in model.J) == d.iloc[i, t]
    
    optimization_model.energy_conservation = pyo.Constraint(
        optimization_model.J,
        optimization_model.T,
        rule = energy_conservation_constraint
    )

    optimization_model.mass_conservation = pyo.Constraint(
        optimization_model.I,
        optimization_model.T,
        rule = mass_conservation_constraint
    )

    def objective(model):
        return (1 / nt) * sum(
            model.h[i, t] - z[i] 
            for i in model.I 
            for t in model.T
        )

    optimization_model.OBJ = pyo.Objective(rule = objective)

    solver = pyo.SolverFactory('ipopt')
    solver.options['tol'] = 1e-3
    solver.options['max_iter'] = 1000
    solver.options['print_level'] = 5
    solver.options['mu_strategy'] = 'adaptive'
    solver.options['mu_oracle'] = 'quality-function'
    solver.options['warm_start_init_point'] = 'yes'

    result = solver.solve(optimization_model, tee=True)
    print(result)


# flow sensor demand data is in m^3/s and time horizon is a week.
flow_data_path = Path("data/sample_field_data/flow.csv")
flow_readings = load_sensor_readings(flow_data_path)

pressure_data_path = Path("data/sample_field_data/pressure.csv")
pressure_readings = load_sensor_readings(pressure_data_path)

flow_map_path = Path("data/network/flow_balance_wwmd.json")
# `flow_balance` maps nodes between WWMD regions to the nodes that flow
# in and out. it provides us the network structure that allows us to 
# calculate mass balance between the regions and ultimately scale demand.
flow_map = json.load(
    open(flow_map_path)
)

epanet_id_lookup_table_path = Path(
    "data/network/InfoWorks_to_EPANET_mapping.xlsx"
)
epanet_id_lookup_table = pd.read_excel(
    epanet_id_lookup_table_path,
    sheet_name = "Nodes",
    dtype = { 'WWMD ID': str }
)

# model demand is in m^3/s and time horizon is a day.
# EPANET models the outflow of reservoir nodes to the adjacent link as demand 
# (confirmed by analysing model in EPANET app). here we only care about
# consumer demand so we'll remove the reservoir nodes.
model_demand_table = simulation_results.node['demand'].drop(
    model.reservoir_name_list,
    axis = 'columns'
)
reservoir_indexes = epanet_id_lookup_table[
    epanet_id_lookup_table["EPANET Node ID"].isin(model.reservoir_name_list)
].index

# TODO: use `nt` consistently throughout implementation.
nt = model.options.time.duration / model.options.time.hydraulic_timestep

d = scale_demands(
    model_demand_table, 
    flow_map, 
    flow_readings, 
    epanet_id_lookup_table.drop(reservoir_indexes)
)

h0 = create_h0(
    pressure_readings,
    epanet_id_lookup_table
)

# time horizon should match model: 1 day
optimization_parameters = OptimizationParameters(
    h0 = h0,
    d = d
)

print(optimization_parameters)

incident_matrixes = IncidentMatrices.from_model(model = model)
print(incident_matrixes)

z = create_z(model = model)
print(z)

# flow_initial?
q_0 = simulation_results.link['flowrate']
# head_initial?
h_0 = simulation_results.node['head']
print(q_0)
print(h_0)

# TODO: is the `HeadlossCoefficients` struct really a good choice here? 
# probably better to use a dataframe.
link_headloss_coefficients = np.zeros(model.num_links, dtype = np.object_)
for i, (_, link) in enumerate(model.links()):
    link_headloss_coefficients[i] = HeadlossCoefficients.from_link(
        link = link, 
        model = model
    )

# TODO: store prev. info on OptimizationParameters struct

# TODO: add AZP weights

# TODO: many warnings that vars are being set beyond their bounds
optimize(
    model = model, 
    nt = nt, 
    z = z, 
    d = d, 
    h0 = h0, 
    A12 = incident_matrixes.A12, 
    A10 = incident_matrixes.A10, 
    link_headloss_coefficients = link_headloss_coefficients
)