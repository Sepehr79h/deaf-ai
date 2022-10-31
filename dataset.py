import numpy as np
import torch, pandas, os, collections
from torch.utils.data import TensorDataset, DataLoader,Dataset, WeightedRandomSampler
from tqdm import tqdm
import dask.array as da
from math import floor

class DataCreator:
    def __init__(self, config_dict):
        self.config_dict = config_dict
        self.batch_size = self.config_dict["batch_size"]

    def create_sampler(self, dataset):
        targets = dataset[:][1]
        class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
        weights = 1. / class_sample_count
        samples_weights = weights[targets]
        assert len(samples_weights) == len(targets)
        return torch.utils.data.sampler.WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)

    def create_loaders(self):
        self.create_x_y()
        # tensor_x = torch.from_numpy(self.X) # transform to torch tensor
        # tensor_y = torch.from_numpy(self.y)

        my_dataset = LargeLoaderDS(self.X,self.y) # create your datset
        train_set,test_set, val_set = torch.utils.data.random_split(my_dataset, [floor(self.y.shape[0]*.7),
            floor(self.y.shape[0]*0),self.y.shape[0]-floor(self.y.shape[0]*.7)-floor(self.y.shape[0]*0)])
        train_sampler = self.create_sampler(train_set)
        # val_sampler = self.create_sampler(val_set)
        # test_sampler = self.create_sampler(test_set)

        train_loader = DataLoader(train_set, batch_size=self.batch_size, sampler=train_sampler)
        val_loader = DataLoader(val_set, batch_size=self.batch_size)#, sampler=val_sampler)
        test_loader = DataLoader(test_set, batch_size=self.batch_size)#, sampler=test_sampler)
        return train_loader, val_loader, test_loader

    def save_files(self, location):
        X = None
        y = []
        for path, subdirs, files in tqdm(os.walk(location)):
            for file in tqdm(files):
                if file.endswith("npy"):

                    curr_array = np.load(os.path.join(path, file), allow_pickle=True)
                    X = curr_array if X is None else np.concatenate((X, curr_array), axis=0)
                    y += [0 if "no_sign_npy" in path else 1]

                    if path.contains("no_sign"):
                        for i in range(0,9):
                            X = curr_array if X is None else np.concatenate((X, curr_array), axis=0)
                            y += [0 if "no_sign_npy" in path else 1]

        np.save("X.npy", X)
        np.save("y.npy", y)

    def create_x_y(self, location='prep_data/'):
        sign = da.from_npy_stack('prep_data/sign_npy', mmap_mode='r')[:100]
        no_sign = da.from_npy_stack('prep_data/no_sign_npy', mmap_mode='r')

        X = da.concatenate((sign,no_sign),axis=0)

        y = np.array([1]*sign.shape[0] + [0]*no_sign.shape[0])

        # if "X.npy" not in os.listdir() or "Y.npy" not in os.listdir() or self.config_dict["override_x_y"] is True:
        #     self.save_files(location)
        #
        # X = np.load("X.npy")
        # y = np.load("y.npy")


        #
        # X = None
        # sign_filenames = [os.path.join("sign_npy", filename) for filename in os.listdir(os.path.join(location, "sign_npy"))]
        # no_sign_filenames = [os.path.join("no_sign_npy", filename) for filename in os.listdir(os.path.join(location, "no_sign_npy")) if filename.endswith("npy")]
        # for filename in sign_filenames + no_sign_filenames:
        #     curr_array = np.load(os.path.join(location, filename))
        #     if X is None:
        #         X = curr_array
        #     else:
        #         X = np.concatenate((X, curr_array), axis=0)
        # y = np.array([0]*len(no_sign_filenames) + [1]*len(sign_filenames))
        print(X.shape, y.shape)
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
        #data = self.xarray[item,:,:,:].values
        data = np.array(self.xarray[item,:,:,:])
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