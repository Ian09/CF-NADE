import random as rd
import numpy as np

"""
Use case
    myData = Data('../ml-1m/ratings.dat')
    myData.split(0.8)
    
    # while training
    X, Y, _, _ = myData.get_batch_train(512, 20)
    
    # after an epoch
    myData.shuffle_train()
    myData.renew_train()
    
"""


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
        X, Y = self.construct_matrix(self.train_users[self.used_train: self.used_train+batch_size], sequence_len, batch_size)
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

    def rating_coding(self, rating_list, seq_len):
        """
        The rating_list is already in time order
        return a 1 * seq_len matrix
        """
        output = np.ones((1, seq_len)) * (-1)
        if len(rating_list) > seq_len:
            rating_list = rating_list[len(rating_list) - seq_len:]

        for i in range(seq_len - len(rating_list), seq_len):
            output[0, i] = self.movie2ratingid(rating_list[i-(seq_len - len(rating_list))][0], rating_list[i- (seq_len - len(rating_list))][1], self.movieDim)
        return output

    def movie2ratingid(self, movie_id, rating, number_movie):
        return (rating - 1) * number_movie + self.movieID2index[movie_id]

    def left2vector(self, ratings):
        """Return a 1 * movie matrix"""
        output = np.zeros((1, self.movieDim))
        for rating in ratings:
            output[0, self.movieID2index[rating[0]]] = rating[1]
        return output

    def construct_matrix(self, users, seq_len, batch_size):
        """
        :param users: list of keys in self.userList(Dict), which contains all triples of certain user
        :param seq_len:
        :return: X(batch * seq), Y(batch * movie)
        """
        matrix_X = np.zeros((batch_size, seq_len))
        matrix_Y = np.zeros((batch_size, self.movieDim))
        i = 0
        for user in users:
            splitPoint = rd.randint(1, len(self.userList[user]) - 1)
            matrix_X[i] = self.rating_coding(self.userList[user][0: splitPoint-1], seq_len)
            matrix_Y[i] = self.left2vector(self.userList[user][splitPoint-1:])
            i += 1

        return matrix_X, matrix_Y


if __name__ == '__main__':

    myData = Data('../ml-1m/ratings.dat')
    myData.split(0.8)
    X, Y, _, _ = myData.get_batch_train(512, 20)
    myData.shuffle_train()
    myData.renew_train()
    X, Y, _, _ = myData.get_batch_train(512, 20)





