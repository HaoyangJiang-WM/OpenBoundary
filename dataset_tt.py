import os
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm


class LamaHDataset(Dataset):
    Q_COL = "qobs"

    def __init__(self, root_dir, years=[2000], root_gauge_id=399, rewire_graph=True,
                 input_window_size=24, target_window_size=6, stride_length=1,
                 total_target_window=6, normalized=True, max_timesteps_per_year=8760):
        """
        Initialize the LamaH dataset.
        Args:
            root_dir (str): Root directory path of the dataset.
            years (list): Years to load, default is [2000].
            root_gauge_id (int): ID of the root gauge.
            rewire_graph (bool): Whether to rewire the graph.
            input_window_size (int): Size of the input window.
            target_window_size (int): Size of the target window for each prediction.
            total_target_window (int): Total size of the target prediction window.
            stride_length (int): Stride length for time steps.
            normalized (bool): Whether to normalize the data.
            max_timesteps_per_year (int): Maximum timesteps per year, default is 8760.
        """
        self.years = years
        self.root_gauge_id = root_gauge_id
        self.rewire_graph = rewire_graph
        self.input_window_size = input_window_size
        self.target_window_size = target_window_size
        self.total_target_window = total_target_window
        self.stride_length = stride_length
        self.normalized = normalized
        self.max_timesteps_per_year = max_timesteps_per_year

        super().__init__(root_dir)

        adj_df = pd.read_csv(self.processed_paths[0])
        self.gauges = list(sorted(set(adj_df["ID"]).union(adj_df["NEXTDOWNID"])))
        rev_index = {gauge_id: i for i, gauge_id in enumerate(self.gauges)}
        edge_cols = adj_df[["ID", "NEXTDOWNID"]].applymap(lambda x: rev_index[x])
        self.edge_index = torch.tensor(edge_cols.values.transpose(), dtype=torch.long)
        weight_cols = adj_df[["dist_hdn", "elev_diff", "strm_slope"]]
        self.edge_attr = torch.tensor(weight_cols.values, dtype=torch.float)

        stats_df = pd.read_csv(self.processed_paths[1], index_col="ID")
        self.mean = torch.tensor(stats_df[[f"{col}_mean" for col in [self.Q_COL]]].values, dtype=torch.float)
        self.std = torch.tensor(stats_df[[f"{col}_std" for col in [self.Q_COL]]].values, dtype=torch.float)

        # Calculate sample size for each year, ensuring it does not exceed max_timesteps_per_year - input_window_size - total_target_window
        self.year_sizes = [
            (self.max_timesteps_per_year - (self.input_window_size + self.total_target_window)) // self.stride_length
            for year in years]
        self.year_tensors = [[] for _ in years]
        print(f"Loading dataset for years: {years} into memory...")

        for gauge_id in tqdm(self.gauges):
            q_df = pd.read_csv(f"{self.raw_dir}/{self.raw_file_names[2]}/hourly/ID_{gauge_id}.csv", sep=";",
                               usecols=["YYYY"] + [self.Q_COL])
            if self.normalized:
                q_df[self.Q_COL] = (q_df[self.Q_COL] - stats_df.loc[gauge_id, f"{self.Q_COL}_mean"]) / stats_df.loc[
                    gauge_id, f"{self.Q_COL}_std"]
            for i, year in enumerate(years):
                q_tensor = torch.tensor(q_df[q_df["YYYY"] == year][self.Q_COL].values, dtype=torch.float).unsqueeze(-1)
                self.year_tensors[i].append(q_tensor)
        self.year_tensors[:] = map(torch.stack, self.year_tensors)

    @property
    def raw_file_names(self):
        return ["B_basins_intermediate_all/1_attributes",
                "B_basins_intermediate_all/2_timeseries",
                "D_gauges/2_timeseries"]

    @property
    def processed_file_names(self):
        return [f"adjacency_{self.root_gauge_id}_{self.rewire_graph}.csv",
                f"statistics_{self.root_gauge_id}_{self.rewire_graph}.csv"]

    def len(self):
        # Dynamically calculate the valid sample length
        valid_sample_length = self.max_timesteps_per_year - (self.input_window_size + self.total_target_window)
        return min(sum(self.year_sizes), valid_sample_length)

    def get(self, idx):
        """
        Get a sample, returning input and target time steps.
        Args:
            idx: Sample index
        Returns:
            Data object containing input x and target y
        """
        year_tensor, offset = self._decode_index(idx)

        # Get input x, selecting only flow data (qobs), shape [num_nodes, input_window_size, 1]
        x = year_tensor[:, offset:(offset + self.input_window_size), 0].unsqueeze(-1)

        # Get target y, ensuring the length of target y is total_target_window, shape [num_nodes, total_target_window, 1]
        y = year_tensor[:,
            (offset + self.input_window_size):(offset + self.input_window_size + self.total_target_window),
            0].unsqueeze(-1)

        # Check if the shape of y is correct
        assert y.size(
            1) == self.total_target_window, f"Target y expected to have window size {self.total_target_window}, but got {y.size(1)}"

        return Data(x=x, y=y, edge_index=self.edge_index, edge_attr=self.edge_attr)

    def _decode_index(self, idx):
        """
        Decode the global index into a specific year's sample and its offset within that year.
        Args:
            idx: Global index
        Returns:
            Data tensor for the corresponding year and the offset
        """
        for i, size in enumerate(self.year_sizes):
            idx -= size
            if idx < 0:
                return self.year_tensors[i], self.stride_length * (idx + size)
        raise AssertionError("Corrupt internal state. This should never happen!")

    def normalize(self, x):
        """
        Normalize the input.
        Args:
            x: Tensor to be normalized
        Returns:
            Normalized tensor
        """
        return (x - self.mean[:, None, :]) / self.std[:, None, :]

    def denormalize(self, x):
        """
        Restore the normalized tensor to the original scale.
        Args:
            x: Normalized tensor to be restored
        Returns:
            Restored tensor
        """
        return self.std[:, None, :] * x + self.mean[:, None, :]


DATASET_PATH = "./LamaH-CE"
dataset = LamaHDataset(DATASET_PATH, years=[2000])
print(dataset[0])
print(dataset.edge_attr.shape)