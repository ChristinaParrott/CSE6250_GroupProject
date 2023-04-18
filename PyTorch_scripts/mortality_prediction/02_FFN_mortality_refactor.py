########################################################################################
#    Citation - reused with modification from:
#
#    Title: patient_trajectory_prediction
#    Author: JamilProg
#    Date: 10/23/2020
#    Availability: https://github.com/JamilProg/patient_trajectory_prediction/blob/master/PyTorch_scripts/mortality_prediction/02_FFN_mortality.py
#
########################################################################################

import pickle
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as dt
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import roc_auc_score as roc
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import gc

RANDOM_SEED = 6250
INPUT_DATA = 'data/prepared_data.npz'
INPUT_DATA2 = 'data/prepared_data_deathTime.npz'
VERBOSE = True
LOG_INTERVAL = 2 # The freqeuncy to output loss data to the console

# HYPERPARAMETERS
HIDDEN_DIM_SIZE = 50
BATCH_SIZE = 50
N_EPOCHS = 50
LEARNING_RATE = 0.001
DROP_OUT = 0.5
K_FOLDS = 3
WITH_CCS = False

class MyNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM_SIZE, drop_out=DROP_OUT, activation=nn.ReLU):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 3)
        self.activation = activation()
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class my_dataset(dt.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


def load_tensors():
    subject_adm_map = pickle.load(open(INPUT_DATA, 'rb'))
    subject_death_map = pickle.load(open(INPUT_DATA2, 'rb'))

    cui_set, ccs_set = set(), set()
    cui_to_int, ccs_to_int = {}, {}

    for subject in subject_adm_map.keys():
        patient_data = subject_adm_map[subject]
        for ith_adm in patient_data:
            for cui_code in ith_adm[2]:
                cui_set.add(cui_code)
            for ccs_code in ith_adm[3]:
                ccs_set.add(ccs_code)
    for i, cui in enumerate(cui_set):
        cui_to_int[cui] = i
    for i, ccs in enumerate(ccs_set):
        ccs_to_int[ccs] = i

    if VERBOSE:
        print(f"{len(subject_adm_map)} patients CUI notes and CCS codes at dimension 0 for file: {INPUT_DATA}")

    num_cui_ints = len(cui_set)
    num_ccs_ints = len(ccs_set)

    if VERBOSE:
        print(f'Remaining patients: {len(subject_adm_map)}')

    vectors_train_x, mortality_train_y = [], []

    for patient_id, adm_list in subject_adm_map.items():
        for i, adm in enumerate(adm_list):
            one_hot_cui = [0] * num_cui_ints
            one_hot_ccs = [0] * num_ccs_ints

            for cui_int in adm[2]:
                one_hot_cui[cui_to_int[cui_int]] = 1
            for ccs_int in adm[3]:
                one_hot_ccs[ccs_to_int[ccs_int]] = 1

            one_hot_x = one_hot_cui + one_hot_ccs if WITH_CCS else one_hot_cui
            vectors_train_x.append(one_hot_x)

            y_sample = [0] * 3
            if (subject_death_map[patient_id] - adm[4]).days <= 0:
                y_sample[0] = 1
            elif (subject_death_map[patient_id] - adm[4]).days <= 30:
                y_sample[1] = 1
            else:
                y_sample[2] = 1
            mortality_train_y.append(y_sample)

    map_index_position = list(zip(vectors_train_x, mortality_train_y))
    random.shuffle(map_index_position)
    vectors_train_x, mortality_train_y = zip(*map_index_position)

    input_dim = num_ccs_ints + num_cui_ints if WITH_CCS else num_cui_ints
    return vectors_train_x, mortality_train_y, input_dim

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def to_byte_tensor(data):
    return torch.ByteTensor(np.concatenate([np.array(plist, dtype=np.uint8) for plist in data]).tolist())

def train():
    X, y, input_dim = load_tensors()
    X_list = list(split(X, 5))
    y_list = list(split(y, 5))

    if VERBOSE:
        print(f"Data of size: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(X_list, y_list, test_size=0.2, random_state=RANDOM_SEED)

    X_train_t = to_byte_tensor(X_train)
    X_test_t = to_byte_tensor(X_test)
    y_train_t = to_byte_tensor(y_train)
    y_test_t = to_byte_tensor(y_test)

    if VERBOSE:
        print(f"Shape X Train: {X_train_t.size()} Shape Y Train: {y_train_t.size()}")
        print(f"Shape X Test: {X_test_t.size()} Shape Y Test: {y_test_t.size()}")


    model = NeuralNetClassifier(
        module=MyNeuralNetwork,
        max_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        criterion=nn.BCEWithLogitsLoss,
        module__input_dim=input_dim
    )

    # param_grid = {
    #     'optimizer': [optim.SGD, optim.Adam, optim.Adamax],
    #     'optimizer__lr': [0.001, 0.01, 0.1, 0.2, 0.3],
    #     'optimizer__momentum': [0.0, 0.2, 0.4, 0.6, 0.8, 0.9],
    #     'module__activation': [nn.ReLU, nn.ReLU6, nn.Tanh, nn.Sigmoid],
    #     'module__dropout': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #     'module__hidden_dim': [1, 5, 10, 15, 20, 25, 30]
    # }

    param_grid = {
        'module__activation': [nn.ReLU, nn.Tanh, nn.Sigmoid],
        'module__drop_out': [0.0, 0.5, 0.8],
        'module__hidden_dim': [25, 50, 100]
    }

    grid = GridSearchCV(estimator=model,
                        param_grid=param_grid,
                        n_jobs=1,
                        cv=K_FOLDS,
                        refit=True,
                        verbose=True)

    grid_result = grid.fit(X_train_t, y_train_t)

    if VERBOSE:
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    if VERBOSE:
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

    best_model = grid_result.best_estimator_
    y_pred = best_model.predict(X_test_t)
    roc_avg_score = roc(y_test_t, y_pred, average='macro', multi_class='ovo')
    if VERBOSE:
        print(f"ROC Average Score:{roc_avg_score}")

    # auc_folds = []
    # for ind_i in range(0, K_FOLDS):
    #     X_test = x_split[ind_i]
    #     Y_test = y_split[ind_i]
    #
    #     split_x_cpy = list(x_split)
    #     split_y_cpy = list(y_split)
    #
    #     del split_x_cpy[ind_i]
    #     del split_y_cpy[ind_i]
    #
    #     x_train = split_x_cpy
    #     y_train = split_y_cpy
    #
    #     model = Network(input_dim)
    #     criterion = nn.BCEWithLogitsLoss()
    #     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #
    #     tensor_x = torch.ByteTensor(np.concatenate([np.array(plist, dtype=np.uint8) for plist in x_train]).tolist())
    #     tensor_y = torch.ByteTensor(np.concatenate([np.array(plist, dtype=np.uint8) for plist in y_train]).tolist())
    #
    #     if VERBOSE:
    #         print(f"Shape X: {tensor_x.size()} Shape Y: {tensor_y.size()}")
    #
    #     dataset = dt.TensorDataset(tensor_x, tensor_y)
    #     train_loader = dt.DataLoader(
    #         dataset,
    #         batch_size=BATCH_SIZE,
    #         shuffle=True)
    #
    #     gc.collect()
    #
    #     losses = []
    #     for epoch in range(N_EPOCHS):
    #         for batch_idx, (data, target) in enumerate(train_loader):
    #             data, target = Variable(data.float()), Variable(target.float())
    #             optimizer.zero_grad()
    #             net_out = model(data)
    #             loss = criterion(net_out, target)
    #             loss.backward()
    #             optimizer.step()
    #             if batch_idx % LOG_INTERVAL == 0:
    #                 if VERBOSE:
    #                     print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx/len(train_loader)}%)]')
    #                     print(f'Loss: {loss.data}')
    #         losses.append(loss.item())
    #
    #     del dataset
    #     del train_loader
    #     del tensor_x
    #     del tensor_y
    #     del split_x_cpy
    #     del split_y_cpy
    #     gc.collect()
    #
    #     tensor_x = torch.Tensor(np.array(X_test).tolist())
    #     tensor_y = torch.Tensor(np.array(Y_test).tolist())
    #     dataset = dt.TensorDataset(tensor_x, tensor_y)
    #     test_loader = dt.DataLoader(
    #         dataset,
    #         batch_size=BATCH_SIZE,
    #         shuffle=False)
    #
    #     model.eval()
    #
    #     y_true = None
    #     y_prob = None
    #     for (data, targets) in test_loader:
    #         x, labels = Variable(data.float()), Variable(targets.float())
    #         outputs = model(x)
    #         outputs = outputs.detach().cpu().numpy()
    #         x = x.detach().cpu().numpy()
    #         labels = labels.detach().cpu().numpy()
    #         for batch_x, batch_true, batch_prob in zip(x, labels, outputs):
    #             y_true = np.concatenate((y_true, [batch_true]), axis=0) if y_true is not None else [batch_true]
    #             y_prob = np.concatenate((y_prob, [batch_prob]), axis=0) if y_prob is not None else [batch_prob]
    #
    #     try:
    #         roc_avg_score = roc(y_true, y_prob, average='macro', multi_class='ovo')
    #         if VERBOSE:
    #             print(f"ROC Average Score:{roc_avg_score}")
    #         auc_folds.append(roc_avg_score)
    #
    #     except:
    #         if VERBOSE:
    #             print("ROC was undefined because classes are all the same")
    #
    # plot_loss(losses)
    #
    # print(f"AUC_scores: {auc_folds}")
    # print(f"Mean: {sum(auc_folds) / len(auc_folds)}")

def plot_loss(train_loss):
    plt.plot(train_loss, label='Training Loss')
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("training_loss.png")

if __name__ == '__main__':
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    train()
