import sys

import torch

sys.path.append("..")
import os
import pandas as pd
import numpy as np
from utils.Utils import Utils
from sklearn.model_selection import train_test_split


class DataPartition:
    def __init__(self, csv_path, split_size, seed=1234,num_users=4):
        self.csv_path = csv_path
        self.split_size = split_size
        self.num_users = num_users
        self.seed=seed

    def addindex(self):
        df = pd.read_csv(self.csv_path, header=None)
        idx = np.arange(len(df))

        df.insert(loc=0, column=None, value=idx)
        df.to_csv('ihdp_new.csv', index=None, header=None)

    def verticalParti(self):
        df = pd.read_csv(self.csv_path, header=None)
        np_ID, np_covariates_X, np_treatment_Y, np_outcome_Y = self.__convert_to_numpy(df)
        num_features = int(np_covariates_X.shape[1] / self.num_users)
        all_idxs = [i for i in range(np_covariates_X.shape[1])]
        result = []
        for i in range(self.num_users):
            t = set(np.random.choice(all_idxs, num_features, replace=False))
            result.append(list(t))
            all_idxs = list(set(all_idxs) - t)
        i = 0
        print(result)
        for item in result:
            i += 1
            vertix = self.concatnumpy(item, np_covariates_X)
            vertix = np.concatenate((np_ID, vertix), axis=1)
            if i == 2:
                vertix = np.concatenate((vertix, np_treatment_Y,np_outcome_Y), axis=1)
            vertidf = pd.DataFrame(vertix)
            vertidf.to_csv("client{0}.csv".format(i), header=False, index=False)

    def concatnumpy(self, rlist, npx):
        nlist = []
        for i in rlist:
            if i == 24:
                nlist.append(npx[:, i:])
            else:
                nlist.append(npx[:, i:i + 1])

        return np.concatenate(tuple(nlist), axis=1)

    def preprocess_data_from_csv(self, csv_path, server_flag):
        print(".. Data Loading ..")
        # data load
        df = pd.read_csv(csv_path, header=None)

        if server_flag:
            np_ID = df.iloc[:, 0:1].to_numpy()
            covariates_X = df.iloc[:, 1:7].to_numpy()
            treatment_Y = df.iloc[:, 7:8].to_numpy()
            outcome_Y = df.iloc[:, 8:10].to_numpy()
            mu = df.iloc[:, 10:12].to_numpy()
            return np_ID, covariates_X, treatment_Y, outcome_Y, mu
        else:
            np_ID = df.iloc[:, 0:1].to_numpy()
            covariates_X = df.iloc[:, 1:8].to_numpy()
            return np_ID, covariates_X

    def convertToTensor(self, np):
        tensor = torch.from_numpy(np).float()
        return tensor

    """
    input:csv,severflag
    output:dict:train set and test set
           0:id 1:x 2:treatment 
    """

    def getDragonNetTensor(self, csv_path, server_flag):
        if server_flag:
            np_ID, covariates_X, treatment_Y, outcome_Y, mu = self.preprocess_data_from_csv(csv_path, server_flag)
            yt = np.concatenate([outcome_Y, treatment_Y], 1)
            id_train, id_test, x_train, x_test, yt_train, yt_test, mu_train, mu_test = train_test_split(np_ID,
                                                                                                        covariates_X,
                                                                                                        yt, mu,
                                                                                                        random_state=self.seed,
                                                                                                        train_size=0.8)
            tensor_id_train = self.convertToTensor(id_train)
            tensor_id_test = self.convertToTensor(id_test)
            tensor_x_train = self.convertToTensor(x_train)
            tensor_x_test = self.convertToTensor(x_test)
            tensor_yt_train = self.convertToTensor(yt_train)
            tensor_yt_test = self.convertToTensor(yt_test)
            tensor_mu_train = self.convertToTensor(mu_train)
            tensor_mu_test = self.convertToTensor(mu_test)
            # return tensor_id_train, tensor_id_test, tensor_x_train, tensor_x_test, t_y_train, t_y_test
            return {"DragonNet_trainData": (tensor_id_train, tensor_x_train, tensor_yt_train, tensor_mu_train),
                    "DragonNet_testData": (tensor_id_test, tensor_x_test, tensor_yt_test, tensor_mu_test)}
        else:
            np_ID, covariates_X = self.preprocess_data_from_csv(csv_path, server_flag)
            id_train, id_test, x_train, x_test = train_test_split(np_ID, covariates_X,
                                                                  random_state=self.seed,
                                                                  train_size=0.8)
            tensor_id_train = self.convertToTensor(id_train)
            tensor_id_test = self.convertToTensor(id_test)
            tensor_x_train = self.convertToTensor(x_train)
            tensor_x_test = self.convertToTensor(x_test)

            return {"DragonNet_trainData": (tensor_id_train, tensor_x_train),
                    "DragonNet_testData": (tensor_id_test, tensor_x_test)}


#
# if __name__ == "__main__":
#     csv_path = './Dataset/ihdp_new.csv'
#     dp = DataPartition(csv_path, 0.8, 4)
#     # dp.verticalParti()
#     id, x, t, o = dp.preprocess_data_from_csv('client2.csv', True)

