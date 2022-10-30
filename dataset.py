import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import dask.array as da
from math import floor

class DataCreator:
    def __init__(self, config_dict):
        self.config_dict = config_dict

    def create_loaders(self):

        # tensor_x = torch.from_numpy(self.X) # transform to torch tensor
        # tensor_y = torch.from_numpy(self.y)

        my_dataset = LargeLoaderDS(self.X,self.y) # create your datset

        train_set,test_set, val_set = torch.utils.data.random_split(my_dataset, [floor(self.y.shape[0]*.7), 
            floor(self.y.shape[0]*.2),self.y.shape[0]-floor(self.y.shape[0]*.7)-floor(self.y.shape[0]*.2)])
        
        train_loader = DataLoader(train_set)
        val_loader = DataLoader(val_set)
        test_loader = DataLoader(test_set)
        return train_loader, val_loader, test_loader
    
    def create_x_y(self, location='prep_data/'):
        sign = da.from_npy_stack('prep_data/sign_npy/')
        no_sign = da.from_npy_stack('prep_data/no_sign_npy/')

        X = da.concatenate((sign,no_sign),axis=0)

        y = np.array([1]*sign.shape[0] + [0]*no_sign.shape[0])

        self.X = X
        self.y = y
        return X,y

class LargeLoaderDS(Dataset):
    def __init__(self, xarray, labels):
        self.xarray = xarray
        self.labels = labels
        self.count = 0
        self.max = xarray.shape[0]

    def __getitem__(self, item):
        data = self.xarray[item,:,:,:].values
        labels = self.labels[item]
        return data, labels

    def __len__(self):
        return self.xarray.shape[0]

    def __iter__(self):
        self.count=0
        return self

    def __next__(self):
        if self.count <= self.max:
            result = self.xarray[self.count]
            self.count += 1
            return result
        else:
            raise StopIteration

dc = DataCreator(dict({}))
dc.create_x_y()
dc.create_loaders()