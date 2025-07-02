import pandapower as pp
import pandapower.networks as pn
import numpy as np
import pandas as pd
from importlib import resources
import scipy.io as spio
from pandapower import makeYbus_pypower
from pandapower.auxiliary import pandapowerNet
import os
from pandapower.converter import to_ppc
import requests


def load_net_from_pp(grid_name: str) -> pandapowerNet:
    """
    Load network from a case file stored in the pandapower library
    """
    network = getattr(pn, grid_name)()
    return network


def load_net_from_file(grid_name: str) -> pandapowerNet:
    """
    Load network from a matpower file
    """
    file_path = str(resources.files(f"GridDataGen.grids").joinpath(f"{grid_name}.m"))
    network = pp.converter.from_mpc(str(file_path))

    old_bus_indices = network.bus.index
    new_bus_indices = range(len(network.bus))

    # Create a mapping dictionary
    bus_mapping = dict(zip(old_bus_indices, new_bus_indices))

    # Reindex the buses in the network
    pp.reindex_buses(network, bus_mapping)

    return network


def load_net_from_pglib(grid_name: str) -> pp.pandapowerNet:
    """
    Load a power grid network from PGLib, downloading if not locally available.

    Parameters:
    -----------
    grid_name : str
        Name of the grid file (e.g., 'case14', 'case118')

    Returns:
    --------
    pandapowerNet
        Loaded power network configuration
    """
    # Construct file paths
    file_path = str(
        resources.files(f"GridDataGen.grids").joinpath(f"pglib_opf_{grid_name}.m")
    )

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Download file if not exists
    if not os.path.exists(file_path):
        url = f"https://raw.githubusercontent.com/power-grid-lib/pglib-opf/master/pglib_opf_{grid_name}.m"
        response = requests.get(url)
        response.raise_for_status()

        with open(file_path, "wb") as f:
            f.write(response.content)

    # Load network from file
    network = pp.converter.from_mpc(file_path)

    old_bus_indices = network.bus.index
    new_bus_indices = range(len(network.bus))

    # Create a mapping dictionary
    bus_mapping = dict(zip(old_bus_indices, new_bus_indices))

    # Reindex the buses in the network
    pp.reindex_buses(network, bus_mapping)

    return network


def save_node_edge_data(
    net: pandapowerNet,
    node_path: str,
    edge_path: str,
    csv_data: list,
    adjacency_lists: list,
):
    """
    Save generated data to CSV files in the data directory, appending if the files already exist.

    Args:
        net (pandapowerNet): The power network.
        node_path (str): File where node data should be stored
        edge_path (str): File where edge data should be stored
        csv_data (list): Node-level data.
        adjacency_lists (list): Edge-level adjacency lists.
    """
    n_buses = net.bus.shape[0]

    # Determine last scenario index
    last_scenario = -1
    if os.path.exists(node_path):
        existing_df = pd.read_csv(node_path, usecols=["scenario"])
        if not existing_df.empty:
            last_scenario = existing_df["scenario"].iloc[-1]

    # 1. Save node data
    df = pd.DataFrame(csv_data)

    if not df.empty:
        # Shift scenario indices
        scenario_indices = np.repeat(
            range(last_scenario + 1, last_scenario + 1 + (df.shape[0] // n_buses)),
            n_buses
        )
        df.insert(0, "scenario", scenario_indices)
        df.to_csv(node_path, mode="a", header=not os.path.exists(node_path), index=False)
    else:
        print("⚠️ Aucune donnée de nœud à sauvegarder.")

    # 2. Save edge data (adjacency)
    if len(adjacency_lists) == 0:
        print("⚠️ Aucune adjacency list – aucun graphe à sauvegarder.")
        return

    try:
        adj_array = np.concatenate(adjacency_lists)
    except ValueError as e:
        print("❌ Impossible de concaténer adjacency_lists :", e)
        return

    if adj_array.shape[1] == 4:
        adj_df = pd.DataFrame(adj_array, columns=["source", "target", "type", "scenario"])
    elif adj_array.shape[1] == 2:
        adj_df = pd.DataFrame(adj_array, columns=["index1", "index2"])
    else:
        raise ValueError(f"⚠️ Forme inattendue pour adjacency_lists : {adj_array.shape}")

    print(f"[DEBUG] Colonnes du adj_df : {adj_df.columns.tolist()}")

# Cast types si applicable
    if "index1" in adj_df.columns and "index2" in adj_df.columns:
        adj_df[["index1", "index2"]] = adj_df[["index1", "index2"]].astype("int64")

# Ajout de colonne "scenario" si absente
    if "scenario" not in adj_df.columns:
        scenario_indices = np.concatenate([
            np.full(adjacency_lists[i].shape[0], last_scenario + 1 + i, dtype="int64")
            for i in range(len(adjacency_lists))
        ])
        adj_df.insert(0, "scenario", scenario_indices)

# Écriture dans le CSV
    adj_df.to_csv(edge_path, mode="a", header=not os.path.exists(edge_path), index=False)
