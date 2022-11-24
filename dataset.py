import numpy as np, json, torch, pandas, os, collections
from torch.utils.data import TensorDataset, DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
import dask.array as da, time
from math import floor
from helpers import ConfigLoader
import pafy
import cv2 as cv, pafy
import numpy as np
import argparse
from tqdm import tqdm

class PoseDatasetCreator:
    def __init__(self, config_dict):
        self.config_dict = config_dict
        self.batch_size = self.config_dict["batch_size"]
        self.json = json.load(open("data_prep/WLASL_v0.3.json"))
        self.sign_pose_array_created = False
        self.sign_pose_array = None
        self.no_sign_pose_array_created = False
        self.no_sign_pose_array = None

    def create_points_array(self):
        counter = 0
        start = time.time()
        for entry in self.json:
            gloss = entry['gloss']
            instances = entry['instances']

            for inst in instances:
                video_url = inst['url']
                video_id = inst['video_id']

                if 'youtube' not in video_url and 'youtu.be' not in video_url:
                    continue
                all_points = PoseDatasetCreator.generate_pose_data(video_url)
                # if isinstance(all_points, np.ndarray):
                #     if self.pose_array_created is False:
                #         self.pose_array = np.expand_dims(all_points, axis=0)
                #     else:
                #         self.pose_array = np.concatenate((self.pose_array, all_points), axis=0)
                if isinstance(all_points, np.ndarray) and all_points.shape[0]!=0:
                    sign_points = all_points[inst['frame_start'] - 1:inst['frame_end']]
                    empty_entry = np.zeros((1, 100, 38))
                    #breakpoint()
                    empty_entry[0, 0:sign_points.shape[0]] = sign_points.copy()[0:min(100, len(sign_points))]
                    # expanded = np.expand_dims(empty_entry, axis=0)
                    if self.sign_pose_array_created is False:
                        self.sign_pose_array = empty_entry
                        self.sign_pose_array_created = True
                        print("Sign Array Created!")
                    else:
                        self.sign_pose_array = np.concatenate((self.sign_pose_array, empty_entry), axis=0)

                    if inst['frame_start'] != 1 or inst['frame_end'] != -1:
                        if vid['frame_start'] != 1:
                            no_sign_points = all_points[0:vid['frame_start'] - 1]
                        elif vid['frame_end'] != -1:
                            no_sign_points = all_points[vid['frame_end'] - 1:-1]
                        empty_entry = np.zeros((1, 100, 38))
                        empty_entry[0, 0:no_sign_points.shape[0]] = no_sign_points.copy()[0:min(100, len(no_sign_points))]
                        if self.no_sign_pose_array_created is False:
                            self.no_sign_pose_array = empty_entry
                            self.no_sign_pose_array_created = True
                            print("No Sign Array Created!")
                        else:
                            self.no_sign_pose_array = np.concatenate((self.no_sign_pose_array, empty_entry), axis=0)

                counter+=1
                sign_print_val = self.sign_pose_array.shape if self.sign_pose_array_created is True else None
                no_sign_print_val = self.no_sign_pose_array.shape if self.no_sign_pose_array_created is True else None
                if counter%10==0:
                    print(counter, time.time() - start, (time.time() - start)/counter)
                    print(f"Sign Array: {sign_print_val}, No Sign Array: {no_sign_print_val}")
                if self.sign_pose_array is not None and self.sign_pose_array.shape[0]>=10:
                    np.save('sign_array.npy', self.sign_pose_array[:6])
                    np.save('no_sign_array.npy', self.sign_pose_array[6:])
                    print(self.sign_pose_array[:6].shape)
        print(counter)
        np.save('sign_array.npy', self.sign_pose_array)
        np.save('no_sign_array.npy', self.no_sign_pose_array)
        breakpoint()


    # create numpy array of wlasl vids that is of shape (num_frames, 38)
    @staticmethod
    def generate_pose_data(video_url, width=368, height=368, max_duration=10):
        body_parts = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                      "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                      "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                      "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

        net = cv.dnn.readNetFromTensorflow("graph_opt.pb")
        all_points = []
        try:
            video = pafy.new(video_url)
            if video.length > max_duration:
                return
            best = video.getbest(preftype="mp4")
            cap = cv.VideoCapture(best.url)
            for i in tqdm(range(0, int(cap.get(cv.CAP_PROP_FRAME_COUNT)))):
                _, frame = cap.read()

                frame_width = frame.shape[1]
                frame_height = frame.shape[0]

                net.setInput(
                    cv.dnn.blobFromImage(frame, 1.0, (width, height), (127.5, 127.5, 127.5), swapRB=True, crop=False))
                out = net.forward()

                points = []
                for j in range(len(body_parts)):
                    # Slice heatmap of corresponding body's part.
                    heat_map = out[0, j, :, :]
                    _, conf, _, point = cv.minMaxLoc(heat_map)
                    x = (frame_width * point[0]) / out.shape[3]
                    y = (frame_height * point[1]) / out.shape[2]
                    # Add a point if it's confidence is higher than threshold.
                    points.append(int(x))
                    points.append(int(y))
                all_points.append(points)

            all_points = np.array(all_points)
        except:
            pass
        return all_points

    def create_sampler(self, dataset):
        targets = dataset[:][1]
        class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
        weights = 1. / class_sample_count
        samples_weights = weights[targets]
        assert len(samples_weights) == len(targets)
        return torch.utils.data.sampler.WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)

    def create_x_y(self, location='prep_data/'):
        sign = np.load("sign_array.npy", allow_pickle=True)
        no_sign = np.load("no_sign_array.npy", allow_pickle=True)

        X = np.concatenate((sign, no_sign), axis=0)
        y = np.array([1] * sign.shape[0] + [0] * no_sign.shape[0])

        print(X.shape, y.shape)
        self.X = X
        self.y = y
        return X, y

    def create_loaders(self):
        self.create_x_y()

        my_dataset = LargeLoaderDS(self.X, self.y)  # create your dataset
        train_set, test_set, val_set = torch.utils.data.random_split(my_dataset, [floor(self.y.shape[0] * .7),
                                                                                  floor(self.y.shape[0] * 0),
                                                                                  self.y.shape[0] - floor(
                                                                                      self.y.shape[0] * .7) - floor(
                                                                                      self.y.shape[0] * 0)])

        train_loader = DataLoader(train_set, batch_size=self.batch_size)
        val_loader = DataLoader(val_set, batch_size=self.batch_size)  # , sampler=val_sampler)
        test_loader = DataLoader(test_set, batch_size=self.batch_size)  # , sampler=test_sampler)
        return train_loader, val_loader, test_loader




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

        my_dataset = LargeLoaderDS(self.X, self.y)  # create your datset
        train_set, test_set, val_set = torch.utils.data.random_split(my_dataset, [floor(self.y.shape[0] * .7),
                                                                                  floor(self.y.shape[0] * 0),
                                                                                  self.y.shape[0] - floor(
                                                                                      self.y.shape[0] * .7) - floor(
                                                                                      self.y.shape[0] * 0)])
        # train_sampler = self.create_sampler(train_set)
        # val_sampler = self.create_sampler(val_set)
        # test_sampler = self.create_sampler(test_set)

        train_loader = DataLoader(train_set, batch_size=self.batch_size)
        val_loader = DataLoader(val_set, batch_size=self.batch_size)  # , sampler=val_sampler)
        test_loader = DataLoader(test_set, batch_size=self.batch_size)  # , sampler=test_sampler)
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
                        for i in range(0, 9):
                            X = curr_array if X is None else np.concatenate((X, curr_array), axis=0)
                            y += [0 if "no_sign_npy" in path else 1]

        np.save("X.npy", X)
        np.save("y.npy", y)

    def create_x_y(self, location='prep_data/'):
        sign = da.from_npy_stack('prep_data/sign_npy', mmap_mode='r')[:100]
        no_sign = da.from_npy_stack('prep_data/no_sign_npy', mmap_mode='r')

        X = da.concatenate((sign, no_sign), axis=0)

        y = np.array([1] * sign.shape[0] + [0] * no_sign.shape[0])

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
        return X, y


class LargeLoaderDS(Dataset):
    def __init__(self, xarray, labels):
        self.xarray = xarray
        self.labels = labels
        self.count = 0
        self.max = xarray.shape[0]

    def __getitem__(self, item):
        # data = self.xarray[item,:,:,:].values
        data = np.array(self.xarray[item, :, :])
        labels = self.labels[item]
        return data, labels

    def __len__(self):
        return self.xarray.shape[0]

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count <= self.max:
            result = self.xarray[self.count]
            self.count += 1
            return result
        else:
            raise StopIteration


if __name__ == "__main__":
    config_dict = ConfigLoader.setup_config("config.yaml")
    pose_dataset_creator = PoseDatasetCreator(config_dict)
    pose_dataset_creator.create_points_array()
