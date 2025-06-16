from dataclasses import dataclass
import datetime as dt
from enum import Enum
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, NewType
import wntr

# assumptions:
# - WWMD resolution is apppopriate (this is more granular than the DMA)

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

WWMD_ID = NewType('WWMD_ID', str)
BWFL_ID = NewType('BWFL_ID', str)
# TODO: update Python to 3.11+ and use StrEnum.
# class FlowMapKey(StrEnum):
#     FLOW_IN = 'flow_in'
#     FLOW_OUT = 'flow_out'
# TODO: update Python to 3.11+ and use StrEnum.
FlowMap = Dict[WWMD_ID, Dict[str, List[BWFL_ID]]]

def scale_demands(
    model_demand_table: pd.DataFrame,
    flow_map: FlowMap, 
    flow_readings: List[SensorReading],
    epanet_id_lookup_table: pd.DataFrame
) -> pd.DataFrame:
    
    """Scales modeled demand for each modelled timestamp according to the 
    relative demand of each WWMD according to live sensors.

    :param model_demand_table: A table describing the modelled demand for each
    EPANET node for each timestep.
    :param flow_map: A map of in/outflow `BWFL_IDs` for WWMDs. Likely decoded
    from JSON.
    :param flow_readings: A list of field sensor readings to use to scale model
    demand. Start time, timestep cadence, and flow units shoud align with ?.
    :param epanet_id_lookup_table: A table describing the `WWMD_ID`s for each
    EPANET node ID. Note there tend to multiple EPANET nodes in a given WWMD.

    :returns: A data frame describing the scaled demands of the nodes (columns)
    for the modelled timestamps.
    """

    scaled_demands = pd.DataFrame(
        data = [],
        index = simulation_results.node['demand'].index
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
        # scaling factor is division of WWMD demand by the modelled demand.
        # note we only take the slice of `sensor_demands` that correspond to
        # `model_demands` as the time horizons may be different. but we assume
        # timesteps are at the same cadence and start times align.
        # TODO: handle division by 0
        scaling_factors = sensor_demands_wwmd.array[:len(model_demand_table_wwmd)] / model_demand_table_wwmd.sum(axis = 1)
        scaled_demands = scaled_demands.join(
            model_demand_table_wwmd.mul(scaling_factors, axis = 'index')
        )
        # sort nodes lexicographically (for easier debugging). 
        # this is a specific command as it can't be achieved via the index-based
        # `join(...)` above.
        scaled_demands.sort_index(axis = 'columns', inplace = True)

    # note some of the modelled demands are 0 (e.g., `node_0004`)
    return scaled_demands
    # TODO: i'm surprised resevoirs (`node_2859`, `node_2860`) actually have 
    # demand -- is this the downstream pipe? note Bradley removed them in his
    # impl.

def load_sensor_readings(path: Path) -> List[SensorReading]:
        # produces a list of dictionaries mapping property/column name to value 
        # for each instance.
        dictionaries = pd.read_csv(
            path,
            parse_dates = True
        ).to_dict(
            orient = "records"
        )
        return [SensorReading(**dictionary) for dictionary in dictionaries]

# pressure_data_path = Path("data/sample_field_data/pressure.csv")
flow_data_path = Path("data/sample_field_data/flow.csv")

# it looks like the {WWMID: "Int, Int"} instances are behaving correctly.
# pressure_readings = load_sensor_readings(pressure_data_path)
flow_readings = load_sensor_readings(flow_data_path)

flow_map_path = Path("data/network/flow_balance_wwmd.json")
# `flow_balance` maps nodes between WWMD regions to the nodes that flow
# in and out. it provides us the network structure that allows us to 
# calculate mass balance between the regions and ultimately scale demand.
flow_map = json.load(
    open(flow_map_path)
)

epanet_id_lookup_table_path = Path("data/network/InfoWorks_to_EPANET_mapping.xlsx")
epanet_id_lookup_table = pd.read_excel(
    epanet_id_lookup_table_path,
    sheet_name = "Nodes",
    dtype={'WWMD ID': str}
)

model_demand_table = simulation_results.node['demand']

# demand is in m^3/s.
# whilst the series are at the same resolution, they don't span the
# same period:
#     - model spans a day (96 entries).
#     - sensor data spans a week.
#
# TODO: i believe Bradley skips the initial state (c.f. 
# optimal-prv-contro.py:135) -- do i need to worry about that?

scale_demands(
    model_demand_table, 
    flow_map, 
    flow_readings, 
    epanet_id_lookup_table
)