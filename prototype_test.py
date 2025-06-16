import datetime as dt
import unittest

import pandas as pd

from prototype import SensorReading, scale_demands

class PrototypeTest(unittest.TestCase):

    def test_scale_demands(self):
        model_demand_table = pd.DataFrame(
            { 
                "epanet_node1": [2, 6],
                "epanet_node2": [1, 2]
            },
            index = [0, 900]
        )
        flow_map = {
            "WWMD_1": {
                "flow_in": ["BWFL_1"],
                "flow_out": ["BWFL_2"]
            }
        }

        dt1 = dt.datetime(2025, 6, 12)
        dt2 = dt.datetime(2025, 6, 12, minute = 15)

        # omitting irrelevant fields for simplicity.
        flow_readings = [
            SensorReading(
                datetime = dt1,
                bwfl_id = "BWFL_1",
                dma_id = "",
                wwmd_id = "",
                min = pd.NA,
                mean = 9,
                max = pd.NA
            ),
            SensorReading(
                datetime = dt1,
                bwfl_id = "BWFL_2",
                dma_id = "",
                wwmd_id = "",
                min = pd.NA,
                mean = 0,
                max = pd.NA
            ),
            SensorReading(
                datetime = dt2,
                bwfl_id = "BWFL_1",
                dma_id = "",
                wwmd_id = "",
                min = pd.NA,
                mean = 8,
                max = pd.NA
            ),
            SensorReading(
                datetime = dt2,
                bwfl_id = "BWFL_2",
                dma_id = "",
                wwmd_id = "",
                min = pd.NA,
                mean = 4,
                max = pd.NA
            ),
        ]
        epanet_id_lookup_table = pd.DataFrame(
            {
                "EPANET Node ID": ["epanet_node1", "epanet_node2"],
                "WWMD ID": ["WWMD_1", "WWMD_1"]
            }
        )
        scaled_demands = scale_demands(
            model_demand_table, 
            flow_map, 
            flow_readings, 
            epanet_id_lookup_table
        )

        # flow is scaled 3x at dt1 (flow difference was 9 compared to 
        # 3 modelled demand)
        # flow is scaled 0.5x at dt2 (flow difference was 4 compared to 
        # 8 modelled demand)

        pd.testing.assert_series_equal(
            scaled_demands["epanet_node1"],
            pd.Series([6.0, 3.0], index = [0, 900], name = "epanet_node1")
        )

        pd.testing.assert_series_equal(
            scaled_demands["epanet_node2"],
            pd.Series([3.0, 1.0], index = [0, 900], name = "epanet_node2")
        )

if __name__ == '__main__':
    unittest.main()

# TODO:
#    - test nil mean
#    - test different combinations of inflow and outflow elements 
# (including empty)    