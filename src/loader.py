from deeppy.data.dataloader.dataloader_base import DataLoaderBase

import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset

import random
from collections import deque, namedtuple
import pickle
import copy



class DatasetLoader(DataLoaderBase):
    """
    Concrete implementation of DatasetLoader for automatic train/test/validation splitting.
    
    This class takes a dataset and automatically splits it into train/test/validation sets
    with configurable ratios. Supports saving/loading split indices for reproducibility.
    
    Features:
    - Automatic data splitting with configurable ratios
    - Custom sampler support (e.g., for data augmentation)
    - Save/load functionality for reproducible splits
    - Handles edge cases (empty splits, small datasets)
    """
    def __init__(self, train_dataset, test_dataset, valid_dataset, 
                train_batch_sampler, test_batch_sampler, valid_batch_sampler,
                splits=[0.8, 0.1, 0.1], file_name=None, 
                 sampler=None, sampler_args=None,
                 batch_size=64, dataloader_args=None):
        """
        Initialize dataset with automatic splitting.
        
        Args:
            data: PyTorch dataset to be split
            splits (list): Ratios for [train, test, valid] splits (must sum to ~1.0)
            file_name (str): If provided, load splits from this path instead of creating new ones
            sampler (class): Custom sampler class for data loading
            sampler_args (dict): Arguments to pass to the sampler
            batch_size (int): Batch size for all data loaders
            dataloader_args (dict): Additional arguments for PyTorch DataLoader
        """
        super().__init__(batch_size=batch_size, dataloader_args=dataloader_args, sampler=sampler)        

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.valid_dataset = valid_dataset
        self.sampler_args = sampler_args or {}

        self.train_batch_sampler = train_batch_sampler
        self.test_batch_sampler = test_batch_sampler
        self.valid_batch_sampler = valid_batch_sampler
        
        self._prepare()
        
    def _prepare(self):
        """Set up datasets and data loaders for all splits."""
        #self._prepare_splits(self.data)
        self.train_loader = self._prepare_dataloader(self.train_dataset, batch_sampler=self.train_batch_sampler)
        self.test_loader = self._prepare_dataloader(self.test_dataset, batch_sampler=self.test_batch_sampler)
        self.valid_loader = self._prepare_dataloader(self.valid_dataset, batch_sampler=self.valid_batch_sampler)

    def _copy_dataset(self, dataset, mode):
        dataset = copy.copy(dataset)
        dataset.init_mode(mode)
        return dataset

    def _prepare_splits(self, data):
        """
        Split the dataset into train/test/validation sets.
        
        Uses random_split with calculated lengths. Handles rounding by giving
        any remainder samples to the training set.
        """
        total_length = len(data)
        lengths = torch.floor(self.splits * total_length).to(torch.int64)
        
        # Add any remainder to training set to ensure all samples are used
        lengths[0] += total_length - lengths.sum()

        #self.train_dataset, self.test_dataset, self.valid_dataset = random_split(
        #    data, lengths.tolist()
        #)
        train, test, valid = random_split(data, lengths.tolist())
        train.dataset = self._copy_dataset(train.dataset, mode='train')
        test.dataset = self._copy_dataset(test.dataset, mode='test')
        valid.dataset = self._copy_dataset(valid.dataset, mode='valid')

        self.train_dataset, self.test_dataset, self.valid_dataset = train, test, valid
    
    def _prepare_dataloader(self, dataset, batch_sampler=None):
        """
        Create a DataLoader for the given dataset split.
        
        Args:
            dataset: PyTorch dataset (could be empty)
            
        Returns:
            DataLoader or None if dataset is empty
        """
        if dataset is None:
            return None
        
        # Ensure batch size doesn't exceed dataset size
        
        # Create dataloader arguments copy to avoid modifying original
        loader_args = self.dataloader_args.copy()
        

        

        return DataLoader(dataset, batch_sampler=batch_sampler, **loader_args)

    def save(self, file_name):
        return
        """
        Save dataset split indices for reproducibility.
        
        Args:
            file_name (str): Path to save split indices (will create split_indices.pkl)
        """
        split_indices = {
            "train": self.train_dataset.indices,
            "test": self.test_dataset.indices,
            "valid": self.valid_dataset.indices
        }
        torch.save(split_indices, f"{file_name}/split_indices.pkl")

    def load(self, data, file_name):
        """
        Load dataset splits from saved indices.
        
        Args:
            data: Original dataset to apply splits to
            file_name (str): Path containing split_indices.pkl
        """
        split_indices = torch.load(f"{file_name}/split_indices.pkl", weights_only=False)
        
        # Recreate dataset splits using saved indices
        train = self._copy_dataset(data, mode='train')
        test = self._copy_dataset(data, mode='test')
        valid = self._copy_dataset(data, mode='valid')
        
        self.train_dataset = Subset(train, split_indices["train"])
        self.test_dataset = Subset(test, split_indices["test"])
        self.valid_dataset = Subset(valid, split_indices["valid"])

        # Create data loaders for all splits
        self.train_loader = self._prepare_dataloader(self.train_dataset, batch_sampler=self.train_batch_sampler)
        self.test_loader = self._prepare_dataloader(self.test_dataset, batch_sampler=self.test_batch_sampler)
        self.valid_loader = self._prepare_dataloader(self.valid_dataset, batch_sampler=self.valid_batch_sampler)

