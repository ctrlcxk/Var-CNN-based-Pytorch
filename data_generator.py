import h5py
import torch
from torch.utils.data import Dataset, DataLoader


class H5Dataset(Dataset):
    def __init__(self, config, data_type, mixture_num):
        """
        Initializes the dataset by loading the HDF5 file and storing the necessary data.

        Args:
            config (dict): Configuration dictionary.
            data_type (str): Either 'training_data', 'validation_data', or 'test_data'.
            mixture_num (int): Index of the mixture to use.
        """
        self.num_mon_sites = config['num_mon_sites']
        self.num_mon_inst_train = config['num_mon_inst_train']
        self.num_mon_inst_test = config['num_mon_inst_test']
        self.num_mon_inst = self.num_mon_inst_train + self.num_mon_inst_test
        self.num_unmon_sites_train = config['num_unmon_sites_train']
        self.num_unmon_sites_test = config['num_unmon_sites_test']
        
        self.data_dir = config['data_dir']
        self.batch_size = config['batch_size']
        self.mixture = config['mixture']
        self.use_dir = 'dir' in self.mixture[mixture_num]
        self.use_time = 'time' in self.mixture[mixture_num]
        self.use_metadata = 'metadata' in self.mixture[mixture_num]

        # Open the HDF5 file and load the necessary datasets
        self.filepath = f"{self.data_dir}{self.num_mon_sites}_{self.num_mon_inst}_{self.num_unmon_sites_train}_{self.num_unmon_sites_test}.h5"
        self.data_type = data_type

        with h5py.File(self.filepath, 'r') as f:
            self.dir_seq = f[f'{self.data_type}/dir_seq'] if self.use_dir else None
            self.time_seq = f[f'{self.data_type}/time_seq'] if self.use_time else None
            self.metadata = f[f'{self.data_type}/metadata'] if self.use_metadata else None
            self.labels = f[f'{self.data_type}/labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Fetches a single data item.

        Args:
            idx (int): Index of the data item.

        Returns:
            tuple: Input dictionary and output dictionary (labels).
        """
        batch_data = {}

        if self.use_dir:
            batch_data['dir_input'] = self.dir_seq[idx]
        if self.use_time:
            batch_data['time_input'] = self.time_seq[idx]
        if self.use_metadata:
            batch_data['metadata_input'] = self.metadata[idx]

        labels = self.labels[idx]
        return batch_data, labels


def create_dataloader(config, data_type, mixture_num, batch_size, shuffle=True, num_workers=4):
    """
    Creates a PyTorch DataLoader.

    Args:
        config (dict): Configuration dictionary.
        data_type (str): Either 'training_data', 'validation_data', or 'test_data'.
        mixture_num (int): Index of the mixture to use.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of worker threads to use for data loading.

    Returns:
        DataLoader: PyTorch DataLoader object.
    """
    dataset = H5Dataset(config, data_type, mixture_num)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
