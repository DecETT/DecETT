import os
import ast
import time
import json
import pickle

import torch

import numpy as np

from tqdm import tqdm
from loguru import logger as logging

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class Dataset(torch.utils.data.Dataset):

    def __init__(self, X, y) -> None:
        super().__init__()

        self.X = X
        self.y = y
        self._len = len(y)
    
    def __len__(self):
        return self._len
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

class DatasetProcessor:

    def __init__(self, args, feature_extractor):
        self.args = args
        self.feature_extractor = feature_extractor

    def yield_sessions(self):
        '''yield data from .log file
        '''
        with open(self.args.path, 'r') as fp:
            for line in fp:
                yield json.loads(line)

    def process(self):
        ''' Extract data from files and attach given labels.
        '''
        sessions = self.yield_sessions()
        
        X, y, S = [], [], []
        t_start = time.time()

        for session in tqdm(sessions):

            pcap_name = session.get('pcap_name')
            src = session.get('src')
            dst = session.get('dst')
            sport = session.get('sport')
            dport = session.get('dport')

            xs, ys = self.feature_extractor(session, self.args)
            for _x, _y in zip(xs, ys):
                X.append(_x)
                y.append(_y)
                S.append([pcap_name, src, dst, sport, dport])
            
            if self.args.max_size > 0 and len(y) > self.args.max_size:
                break

        t_end = time.time()
        preprocess_time = t_end - t_start
        logging.info(f'Process data success! Duration: {preprocess_time}s')
        
        X, y, S = np.array(X), np.array(y), np.array(S)

        if self.args.scaler:
            X = MinMaxScaler().fit_transform(X)
        
        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)

        return X, y, S, preprocess_time

    def split(self, X, y, S, valid_ratio, test_ratio, random_seed):
        ''' Split data into train and test sets.
        '''

        X_indices = np.arange(len(X))
        X_train, X_test, y_train, y_test = train_test_split(X_indices, y, test_size=valid_ratio+test_ratio, random_state=random_seed, stratify=y)
        valid_ratio = valid_ratio / (valid_ratio + test_ratio)
        if valid_ratio == 0.:
            X_valid, y_valid = [], []
        elif valid_ratio == 1.:
            X_valid, y_valid = X_test, y_test
        else:
            X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=valid_ratio, random_state=random_seed, stratify=y_test)

        S_train, S_valid, S_test = S[X_train], S[X_valid], S[X_test]
        X_train, X_valid, X_test = X[X_train], X[X_valid], X[X_test]

        return X_train, X_valid, X_test, y_train, y_valid, y_test, S_train, S_valid, S_test
    
    def save(self, outfile, X, y, S):
        """Save data to given outfile.

            Parameters
            ----------
            outfile : string
                Path of file to save data to.

            X : np.array of shape=(n_samples, n_features)
                Features extracted from files.

            y : np.array of shape=(n_samples,)
                Labels for each flow extracted from files.
            """
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        with open(outfile, 'wb') as fp:
            pickle.dump((X, y, S), fp)

    def load(self, infile):
        """Load data from given infile.

            Parameters
            ----------
            infile : string
                Path of file from which to load data.

            Returns
            -------
            X : np.array of shape=(n_samples, n_features)
                Features extracted from files.

            y : np.array of shape=(n_samples,)
                Labels for each flow extracted from files.
            """
        with open(infile, 'rb') as infile:
            return pickle.load(infile)
    
    def has_save(self, savefile):
        """Check if data has been saved to given file.
        
            Parameters
            ----------
            savefile : string
                Path of savedir to check.

            Returns
            -------
            bool
                True if file exists, False otherwise.
        
        """
        return os.path.exists(savefile)


class DatasetProcessor_DRL(DatasetProcessor):
    def __init__(self, args, feature_extractor, mode=""):
        super().__init__(args, feature_extractor)
        self.args = args
        self.max_packet_len = 0
        self.mode = mode

    def yield_sessions(self):
        if self.mode == "":
            with open(self.args.path, 'r') as fp:
                for line in fp:
                    yield line

        elif self.mode == "pretrain":
            with open(self.args.pretrain_path, 'r') as fp:
                for line in fp:
                    yield line

        elif self.mode == "finetune_joint":
            with open(self.args.old_path, 'r') as fp:
                for line in fp:
                    yield line
    
    def process(self):
        yield_res = self.yield_sessions()

        X_tls, X_tun, _Y, info = [], [], [], []
        
        t_start = time.time()

        for _flow_pair in tqdm(yield_res):
            tls_flow, tun_flow = ast.literal_eval(_flow_pair)

            x_tls, x_tun, y = self.feature_extractor(tls_flow, tun_flow, self.args)
            self.max_packet_len = self.feature_extractor.max_packet_len

            for _x_tls, _x_tun, _y in zip(x_tls, x_tun, y):
                X_tls.append(_x_tls)
                X_tun.append(_x_tun)
                _Y.append(_y)

            if self.args.max_size > 0 and len(y) > self.args.max_size:
                break
        
        sorted_Y = list(set(sorted(_Y)))
        print(sorted_Y)

        label_mapping = {label: idx for idx, label in enumerate(sorted_Y)}
        print(label_mapping)
        Y = [label_mapping[label] for label in _Y]

        t_end = time.time()
        preprocess_time = t_end - t_start
        logging.info(f"Process data success! Duration: {preprocess_time}s")

        X_tls, X_tun, Y = np.array(X_tls), np.array(X_tun), np.array(Y)

        
        min_y = np.min(Y)
        print(min_y)
        if min_y != 0:
            Y = Y - min_y

        torch.set_printoptions(threshold=float('inf'))
        

        if self.args.scaler:
            X_tls = MinMaxScaler().fit_transform(X_tls)
            X_tun = MinMaxScaler().fit_transform(X_tun)
        
        X_tls = torch.tensor(X_tls, dtype=torch.float)
        X_tun = torch.tensor(X_tun, dtype=torch.float)
        Y = torch.tensor(Y, dtype=torch.long)

        X = torch.stack((X_tls, X_tun), dim=1)
        

        return X, Y, info, preprocess_time, label_mapping


    def split(self, X, y, S, valid_ratio, test_ratio, random_seed):

        if test_ratio == 1.0:
            return [], [], X, [], [], y, [], [], []

        X_indices = np.arange(len(X))
        X_train, X_test, y_train, y_test = train_test_split(X_indices, y, test_size=valid_ratio+test_ratio, random_state=random_seed, stratify=y)
        
        valid_ratio = valid_ratio / (valid_ratio + test_ratio)
        if valid_ratio == 0.:
            X_valid, y_valid = [], []
        elif valid_ratio == 1.:
            X_valid, y_valid = X_test, y_test
        else:
            X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=valid_ratio, random_state=random_seed, stratify=y_test)

        X_train, X_valid, X_test = X[X_train], X[X_valid], X[X_test]

        return X_train, X_valid, X_test, y_train, y_valid, y_test, [], [], []
    

    def save(self, outfile, X, y, S, label_mapping):
        """Save data to given outfile.

            Parameters
            ----------
            outfile : string
                Path of file to save data to.

            X : np.array of shape=(n_samples, n_features)
                Features extracted from files.

            y : np.array of shape=(n_samples,)
                Labels for each flow extracted from files.
            """
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        with open(outfile, 'wb') as fp:
            pickle.dump((X, y, S, label_mapping), fp)


    def load(self, infile):
        """Load data from given infile.

            Parameters
            ----------
            infile : string
                Path of file from which to load data.

            Returns
            -------
            X : np.array of shape=(n_samples, n_features)
                Features extracted from files.

            y : np.array of shape=(n_samples,)
                Labels for each flow extracted from files.
            """
        t_start = time.time()
        with open(infile, 'rb') as infile:
            # X, Y, info, label_mapping = pickle.load(infile)
            X, Y = pickle.load(infile)
        t_end = time.time()
        process_time = t_end - t_start
        
        return X, Y, [], process_time, {}
    

    def has_save(self, savefile):
        """Check if data has been saved to given file.
        
            Parameters
            ----------
            savefile : string
                Path of savedir to check.

            Returns
            -------
            bool
                True if file exists, False otherwise.
        
        """
        return os.path.exists(savefile)
