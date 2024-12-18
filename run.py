import numpy as np
import torch
from easydict import EasyDict
import pickle
from trainer import trainer

def circle_points(r, n):
    """
    generate evenly distributed unit divide vectors for two tasks
    """
    circles = []
    for r, n in zip(r, n):
        t = np.linspace(0, 0.5 * np.pi, n)
        x = r * np.cos(t)
        y = r * np.sin(t)
        circles.append(np.c_[x, y])
    return circles



def paretoMLT_exp(config):
    config = EasyDict(config)

    ref_vec = torch.tensor(circle_points([1], [config.num_pref])[0]).float()

    data = pickle.load(open('data/{}/train_val_test_{}.pkl'.format(config.data_name, config.fold), 'rb'))

    config.num_items = data['num_items_Q']
    config.num_users = data['num_users']

    print(config)

    # for i in range(config.num_pref):
    for i in range(2, 3):  # 3th is the middle preference vector
        pref_idx = i
        exp_trainner = trainer(config, data, ref_vec, pref_idx)
        exp_trainner.train()


def ednet():
    for fold in [1, 2, 3, 4, 5]:
        config = {
                    "data_name": 'ednet',
                    "model_name": "Singleview_KTBM",

                    "mode": 'test',
                    "fold": fold,
                    "metric": 'auc',
                    "shuffle": True,

                    "cuda": True,
                    "gpu_device": 0,
                    "seed": 1024,

                    "n_tasks": 2,
                    "num_pref": 5,  # number of dividing vector

                    "min_seq_len": 2,
                    "max_seq_len": 100,  # the max step of RNN model
                    "batch_size": 32,
                    "learning_rate": 0.01,
                    "max_epoch": 70,
                    "validation_split": 0.2,

                    "embedding_size_q": 32,
                    "embedding_size_a": 32,
                    # "embedding_size_l": 32,
                    "num_concepts": 8,
                    "key_dim": 32,
                    "value_dim": 32,
                    "summary_dim": 32,

                    "k_neighbors": 10,
                    "top_k_eva": 5,

                    'weight_material': 0.01,

                    "init_std": 0.2,
                    "max_grad_norm": 10,

                    "optimizer": 'adam',
                    "epsilon": 0.1,
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "weight_decay": 0.05,
                }
        paretoMLT_exp(config)

def Junyi2063():
    for fold in [1, 2, 3, 4, 5]:
        config = {
                "data_name": 'Junyi2063',
                "model_name": 'Singleview_KTBM',

                "mode": 'test',
                "fold": fold,
                "metric": 'auc',
                "shuffle": True,

                "cuda": True,
                "gpu_device": 0,
                "seed": 1024,

                "n_tasks": 2,
                "num_pref": 5,  # number of dividing vector

                "min_seq_len": 2,
                "max_seq_len": 100,  # the max seq len of model
                "batch_size": 32,
                "learning_rate": 0.01,
                "max_epoch": 60,
                "validation_split": 0.2,

                "embedding_size_q": 32,
                "embedding_size_a": 32,
                # "embedding_size_l": 32,
                "num_concepts": 32,
                "key_dim": 64,
                "value_dim": 64,
                "summary_dim": 32,

                "k_neighbors": 10,
                "top_k_eva": 5,

                'weight_material': 0.01,

                "init_std": 0.2,
                "max_grad_norm": 50,

                "optimizer": 'adam',
                "epsilon": 0.1,
                "beta1": 0.9,
                "beta2": 0.999,
                "weight_decay": 0.05,
            }
        paretoMLT_exp(config)



if __name__== '__main__':
    # ednet()
    Junyi2063()