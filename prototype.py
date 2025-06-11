from dataclasses import dataclass
import datetime as dt
import json
import pandas as pd
from pathlib import Path
from typing import List, Optional
import wntr

network_path = Path("data/network/BWFLnet_2023_04.inp")
model = wntr.network.WaterNetworkModel(network_path)
simulator = wntr.sim.EpanetSimulator(model)
simulation_results = simulator.run_sim()

# TODO: should temp.* files go in .gitignore?

# TODO: plot data to better understand demand scaling

# TODO: explore if i'll consider the model calibrated or replace with 
# calibration coefficients (as Bradley did in `bayesian-wq-calibration`).

# TODO: replace with live sensor data. for now i'll either use Bradley's data
# or stochastic data (this will lend itself to writing tests).

# assuming WWMD resolution is apppopriate for our purposes (i believe this
# is more granular than the DMA) 

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
    pressure_data_path = Path("data/sample_field_data/pressure.csv")
    flow_data_path = Path("data/sample_field_data/flow.csv")

    pressure_objects = load_sensor_readings(pressure_data_path)
    flow_objects = load_sensor_readings(flow_data_path)

    # print(pressure_objects)
    # # it looks like the {WWMID: "Int, Int"} instances are behaving correctly.
    # print(flow_objects)
    print(flow_objects[14])

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

scale_demands()