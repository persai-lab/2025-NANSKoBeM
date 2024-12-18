import numpy as np
import torch
import math
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data import TensorDataset
import more_itertools as miter
from collections import defaultdict

class NANS_KoBeM_DataLoader:
    def __init__(self, config, data):
        self.batch_size = config["batch_size"]
        self.shuffle = config["shuffle"]
        self.collate_fn = default_collate
        self.metric = config["metric"]

        self.num_questions = config.num_items
        self.k_neighbors = config.k_neighbors

        self.seed = config['seed']

        self.validation_split = config["validation_split"]
        self.mode = config["mode"]

        self.min_seq_len = config["min_seq_len"] if "min_seq_len" in config else None
        self.max_seq_len = config["max_seq_len"] if "max_seq_len" in config else None
        self.stride = config["max_seq_len"] if "max_seq_len" in config else None

        self.init_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
        }

        self.generate_train_test_data(data)
        self.read_graph_neighbor(data)

        # define the data format for different perfromance data
        if self.metric == 'rmse':
            self.train_data = TensorDataset(torch.Tensor(self.train_data_q).long(),
                                            torch.Tensor(self.train_data_a).float(),
                                            torch.Tensor(self.train_target_answers).float(),
                                            torch.Tensor(self.train_target_masks).bool()
                                            )

            self.test_data = TensorDataset(torch.Tensor(self.test_data_q).long(), torch.Tensor(self.test_data_a).float(),
                                           torch.Tensor(self.test_target_answers).float(),
                                           torch.Tensor(self.test_target_masks).bool(),
                                           torch.Tensor(self.test_data_neg).long()
                                           )

        else:
            self.train_data = TensorDataset(torch.Tensor(self.train_data_q).long(),
                                            torch.Tensor(self.train_data_a).long(),
                                            torch.Tensor(self.train_target_answers).long(),
                                            torch.Tensor(self.train_target_masks).bool()
                                            )


            self.test_data = TensorDataset(torch.Tensor(self.test_data_q).long(), torch.Tensor(self.test_data_a).long(),
                                           torch.Tensor(self.test_target_answers).long(),
                                            torch.Tensor(self.test_target_masks).bool(),
                                           torch.Tensor(self.test_data_neg).long()
                                           )

        # create batched data
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size)

        self.test_loader = DataLoader(self.test_data, batch_size=self.test_data_a.shape[0])



    def generate_train_test_data(self, data):
        """
        read or process data for training and testing
        """

        q_records = data["traindata"]["q_data"]
        a_records = data["traindata"]["a_data"]

        self.train_data_q, self.train_data_a = self.Singleview_KTBM_ExtDataset(q_records,
                                                                                                       a_records,
                                                                                                       self.max_seq_len,
                                                                                                       stride=self.stride)

        self.train_target_answers = np.copy(self.train_data_a)
        self.train_target_masks = (self.train_data_q != 0)

        if self.mode == "train":
            # n_samples = len(self.train_data_q)
            # split the train data into train and val sets based on the self.n_samples

            self.train_data_q, self.test_data_q, self.train_data_a, self.test_data_a,
            self.train_target_answers, self.test_target_answers, \
            self.train_target_masks, self.test_target_masks = train_test_split(
                self.train_data_q, self.train_data_a, self.train_target_answers,
                self.train_target_masks)


        elif self.mode == 'test':
            q_records = data["testdata"]["q_data"]
            a_records = data["testdata"]["a_data"]
            neg_records = data["testdata"]["neg_evaluation_data"]

            self.test_data_q, self.test_data_a, self.test_data_neg = self.Singleview_KTBM_ExtDataset_test(q_records,
                                                                                a_records, neg_records,
                                                                                self.max_seq_len,
                                                                                stride=self.stride)

            self.test_target_answers = np.copy(self.test_data_a)
            self.test_target_masks = (self.test_data_q != 0)


    def Singleview_KTBM_ExtDataset(self, q_records, a_records,
                          max_seq_len,
                          stride):
        """
        transform the data into feasible input of model,
        truncate the seq. if it is too long and
        pad the seq. with 0s if it is too short
        """

        q_data = []
        a_data = []
        for index in range(len(q_records)):
            q_list = q_records[index]
            a_list = a_records[index]

            # if seq length is less than max_seq_len, the windowed will pad it with fillvalue
            # the reason for inserting two padding attempts with 0 and setting stride = stride - 2 is to make sure the
            # first activity of each sequence is included in training and testing, and also for each sequence's first
            # activity there is an activity zero to be t - 1 attempt.

            q_list.insert(0, 0)
            a_list.insert(0, 2)

            q_list.insert(0, 0)
            a_list.insert(0, 2)
            q_patches = list(miter.windowed(q_list, max_seq_len, fillvalue=0, step=stride - 2))
            a_patches = list(miter.windowed(a_list, max_seq_len, fillvalue=2, step=stride - 2))

            q_data.extend(q_patches)
            a_data.extend(a_patches)

        return np.array(q_data), np.array(a_data)



    def Singleview_KTBM_ExtDataset_test(self, q_records, a_records, neg_records,
                          max_seq_len,
                          stride):
        """
        transform the data into feasible input of model,
        truncate the seq. if it is too long and
        pad the seq. with 0s if it is too short
        """

        q_data = []
        a_data = []
        neg_data = []
        for index in range(len(q_records)):
            q_list = q_records[index]
            a_list = a_records[index]
            neg_list = neg_records[index]

            # if seq length is less than max_seq_len, the windowed will pad it with fillvalue
            # the reason for inserting two padding attempts with 0 and setting stride = stride - 2 is to make sure the
            # first activity of each sequence is included in training and testing, and also for each sequence's first
            # activity there is an activity zero to be t - 1 attempt.

            q_list.insert(0, 0)
            a_list.insert(0, 2)
            neg_list.insert(0, [0]*99)

            q_list.insert(0, 0)
            a_list.insert(0, 2)
            neg_list.insert(0, [0]*99)
            q_patches = list(miter.windowed(q_list, max_seq_len, fillvalue=0, step=stride - 2))
            a_patches = list(miter.windowed(a_list, max_seq_len, fillvalue=2, step=stride - 2))
            neg_patches = list(miter.windowed(neg_list, max_seq_len, fillvalue=[0]*99, step=stride - 2))

            q_data.extend(q_patches)
            a_data.extend(a_patches)
            neg_data.extend(neg_patches)

        return np.array(q_data), np.array(a_data), np.array(neg_data)



    def read_graph_neighbor(self, data):
        q_records = self.train_data_q

        # q_q_neighbors = [set() for i in range(self.num_questions + 1)]
        q_q_neighbors = [defaultdict(lambda: 0) for i in range(self.num_questions + 1)]
        # q_q_neighbors_weights = np.zeros([self.num_questions+1, self.num_questions+1])
        for index in range(len(q_records)):
            q_list = q_records[index]

            for att in range(3, len(q_list) - 1):
                if q_list[att] != 0 and q_list[att - 1] != 0:
                    q_q_neighbors[q_list[att]][q_list[att - 1]]+=1
                    q_q_neighbors[q_list[att - 1]][q_list[att]]+=1
            #
            # for att in range(3, len(q_list) - 1):
            #     if q_list[att] != 0 and q_list[att - 1] != 0:
            #         q_q_neighbors_weights[q_list[att]][q_list[att - 1]]+=1
            #         q_q_neighbors_weights[q_list[att - 1]][q_list[att]]+=1

        for i in range(self.num_questions):
            # q_q_neighbors_weights[i, i] = 0
            if i in q_q_neighbors[i]:
                q_q_neighbors[i].pop(i)
                q_q_neighbors[i] = dict(sorted(q_q_neighbors[i].items(), key=lambda item: item[1], reverse=True))

        # q_q_neighbors_weights = q_q_neighbors_weights/np.sum(q_q_neighbors_weights)
        # q_q_neighbors_weights = q_q_neighbors_weights/np.sum(q_q_neighbors_weights, axis=1)
        # self.q_q_neighbors_weights = np.nan_to_num(q_q_neighbors_weights.T)

        q_q_neighbors_list = []
        for i in q_q_neighbors:
            nebrs = list(i.keys())
            # weight = i.values()
            if len(nebrs) == 0:
                q_q_neighbors_list.append([0]*self.k_neighbors)
            elif len(nebrs) >= self.k_neighbors:
                q_q_neighbors_list.append(nebrs[:self.k_neighbors])
            else:
                pad_times = math.ceil(self.k_neighbors/len(nebrs))
                nebrs = nebrs*pad_times
                q_q_neighbors_list.append(nebrs[:self.k_neighbors])

        self.q_q_neighbors = torch.Tensor(q_q_neighbors_list).long()
