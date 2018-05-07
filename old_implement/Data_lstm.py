import random as rd
import numpy as np

class Data:

    def __init__(self, data_directory):
        self.userList = {}
        self.movieID2index = {}
        index = 0
        self.used_train = 0
        self.used_test = 0

        with open(data_directory, 'r') as f:
            for line in f.readlines():
                userID, movieID, rating, timeStamp = line.split('::')
                timeStamp = timeStamp.strip()
                if int(movieID) not in self.movieID2index:
                    self.movieID2index[int(movieID)] = index
                    index += 1
                if userID in self.userList:
                    self.userList[userID].append((int(movieID), int(rating), int(timeStamp)))
                else:
                    self.userList[userID] = [(int(movieID), int(rating), int(timeStamp))]
        self.movieDim = len(self.movieID2index)

        # sort rating triples for each user according to time stamp
        for key in self.userList:
            ratingsTriples = self.userList[key]
            ratingsTriplesSorted = sorted(ratingsTriples, key=lambda x: x[2])
            self.userList[key] = ratingsTriplesSorted
        self.user = len(self.userList)

    def split(self, ratio_train):
        all_users = list(self.userList.keys())
        self.train_users = all_users[0: int(ratio_train * self.user)]
        self.test_users = all_users[int(ratio_train * self.user):]

    def shuffle_train(self):
        rd.shuffle(self.train_users)

    def get_batch_train(self, batch_size, sequence_len):
        """return X(batch * seq), Y(batch * movie), output_mask(batch * movie), flag"""
        if batch_size + self.used_train >= len(self.train_users):
            return None, None, None, False
        X, Y = self.construct_matrix(self.train_users[self.used_train : self.used_train+batch_size], sequence_len)
        output_mask = Y > 0
        self.used_train += batch_size
        return X, Y, output_mask, True

    def get_batch_test(self, batch_size, sequence_len):
        """return X(batch * seq), Y(batch * movie), output_mask(batch * movie), flag"""
        if batch_size + self.used_test >= len(self.test_users):
            return None, None, None, False
        X, Y = self.construct_matrix(self.test_users[self.test_train: self.test_train+batch_size], sequence_len)
        output_mask = Y > 0
        self.used_test += batch_size
        return X, Y, output_mask, True

    def renew_train(self):
        self.used_train = 0
        rd.shuffle(self.train_users)

    def renew_test(self):
        self.used_test = 0
        rd.shuffle(self.test_users)

    def construct_matrix(self, users, seq_len):
        """
        :param users: list of keys in self.userList(Dict), which contains all triples of certain user
        :param seq_len:
        :return: X(batch * seq), Y(batch * movie)
        """
        pass




