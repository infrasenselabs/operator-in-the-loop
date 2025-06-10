import wntr

# assuming this file is at the `ROOT_DIR` of the project.
network_path = "data/network/BWFLnet_2023_04.inp"
model = wntr.network.WaterNetworkModel(network_path)
simulator = wntr.sim.EpanetSimulator(model)
simulation_results = simulator.run_sim()

# TODO: should temp.* files go in .gitignore?
print(simulation_results)
