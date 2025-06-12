from dataclasses import dataclass
import datetime as dt
import json
import pandas as pd
from pathlib import Path
from typing import List, Optional
import wntr

# assumptions:
# - WWMD resolution is apppopriate (this is more granular than the DMA)

network_path = Path("data/network/BWFLnet_2023_04.inp")
model = wntr.network.WaterNetworkModel(network_path)
simulator = wntr.sim.EpanetSimulator(model)
simulation_results = simulator.run_sim()
print(simulation_results)

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

def scale_demands():
    epanet_wwmd_id_map_path = Path("data/network/InfoWorks_to_EPANET_mapping.xlsx")
    epanet_wwmd_id_map = pd.read_excel(epanet_wwmd_id_map_path, sheet_name="Nodes", dtype={'WWMD ID': str})

    scaled_demands = pd.DataFrame(
        data = [],
        index = simulation_results.node['demand'].index
    )

    for wwmd_id, flows in flow_balance.items():
        # TODO: this may be where seeing if it's a probabiltiy distribution
        # could be useful.
        flow_in_series = pd.DataFrame(
            data = [
                reading.__dict__
                for reading in flow_readings 
                if reading.bwfl_id in flows['flow_in']
            ],
            columns = ["datetime", "mean"]
        ).groupby('datetime')["mean"].sum()

        flow_out_series = pd.DataFrame(
            data = [
                reading.__dict__
                for reading in flow_readings 
                if reading.bwfl_id in flows['flow_out']
            ],
            columns = ["datetime", "mean"]
        ).groupby('datetime')["mean"].sum()

        # demand is in m^3/s.
        # find the corresponding EPANET node for the current WWMD ID in
        # the map, filtering the model demands to just that node.
        epanet_node_ids = epanet_wwmd_id_map.loc[
                epanet_wwmd_id_map["WWMD ID"] == wwmd_id
            ]["EPANET Node ID"]
        if len(epanet_node_ids) > 0:
            # whilst the series are at the same resolution, they don't span the
            # same period:
            #     - model spans 1 day (96 entries) and first instance represents
            #       inital state (time = 0).
            #     - sensor data spans a week; first instance also represents
            #       inital state.
            # TODO: i believe Bradley skips the initial state (c.f. 
            # optimal-prv-contro.py:135) -- do i need to worry about that?
            #
            # the demand per timestep for the entire WWMD, as calculated per
            # the sensors.
            sensor_demands = flow_in_series.sub(flow_out_series, fill_value=0).array
            model_demands = simulation_results.node['demand'][epanet_node_ids]
            # scaling factor is division of WWMD demand (as calculated by
            # sensors) by the modeled demands for all nodes in this WWMD,
            # for each timestep.
            scaling_factors = sensor_demands[:len(model_demands)] / model_demands.sum(axis=1)
            scaled_demands = scaled_demands.join(
                model_demands.mul(scaling_factors, axis='index')
            )
            # sort nodes lexicographically (for easier debugging). can't be
            # achieved in index-based (vs. column-based) `join(...)`.
            scaled_demands.sort_index(axis='columns', inplace=True)
        else:
            raise Exception(f"No matching EPANET Node ID for {wwmd_id}")

    # some of the modelled demands are 0 (e.g., `node_0004`)
    # TODO: i'm surprised resevoirs (`node_2859`, `node_2860``) actually have 
    # demand -- is this the downstream pipe? note Bradley removed them in his
    # impl.
    print(scaled_demands)
    print(model.reservoir_name_list)
    print("done")

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

pressure_data_path = Path("data/sample_field_data/pressure.csv")
flow_data_path = Path("data/sample_field_data/flow.csv")

# it looks like the {WWMID: "Int, Int"} instances are behaving correctly.
pressure_readings = load_sensor_readings(pressure_data_path)
flow_readings = load_sensor_readings(flow_data_path)

flow_balance_path = Path("data/network/flow_balance_wwmd.json")
# `flow_balance` maps nodes between WWMD regions to the nodes that flow
# in and out. it provides us the network structure that allows us to 
# calculate mass balance between the regions and ultimately scale demand.
flow_balance = json.load(
        # TODO: error handling
    open(flow_balance_path)
)
scale_demands()