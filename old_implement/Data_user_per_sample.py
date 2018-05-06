import random as rd
import numpy as np
"""
Use case:
    myData = Data_user(data_file_path)
    myData.split_set(0.8, 0.1)  # portion of train set and development set
    myData.prepare_data()
    
    # get train batch
    rating, output_mask, input_mask, flag = myData.get_batch_train(batch_size)  # when epoch ends, it returns (None, None, None, False)
    
    # development RMSE
    rating, output_mask, input_mask, flag = myData.get_batch_dev(batch_size)
    
    # after you train for an epoch
    myData.renew_train()
    myData.renew_dev()
    myData.shuffle_data()
    
    # test RMSE
    rating, output_mask, input_mask, flag = myData.get_batch_test(batch_size)
"""


class Data_user():
    def __init__(self, data_directory):
        self.used_train = 0
        self.used_dev = 0
        self.used_test = 0
        index = 0
        self.movieID2index = {}
        self.userList = {}
        with open(data_directory, 'r') as f:
            for line in f.readlines():
                userID, movieID, rating, timeStamp = line.split('::')
                timeStamp = timeStamp.strip()
                if int(movieID) not in self.movieID2index:
                    self.movieID2index[int(movieID)] = index
                    index += 1

                if userID in self.userList:
                    self.userList[userID].append((int(movieID), int(float(rating)), int(timeStamp)))
                else:
                    self.userList[userID] = [(int(movieID), int(float(rating)), int(timeStamp))]

        self.M = len(self.movieID2index)  # number of movies
        self.user = len(self.userList)  # number of users

        self.all_records = np.zeros((self.user, self.M))  # rating matrix (user by movies)
        self.output_mask = np.ones((self.user, self.M), dtype=bool)
        self.input_mask = np.ones((self.user, self.M), dtype=bool)

    def prepare_data(self):
        i = 0
        all_keys = list(self.userList.keys())
        rd.shuffle(all_keys)
        keys_train_dev, keys_test = all_keys[0: self.num_train + self.num_dev], all_keys[self.num_train + self.num_dev:]
        for key in keys_train_dev:
            ratingsTriples = self.userList[key]
            """ratings before i would be trainset, exclusive, that is [0, i-1]"""
            splitPoint = rd.randint(1, len(ratingsTriples) - 1)
            rd.shuffle(ratingsTriples)
            matrix_rating, input_mask, output_mask = self.triples2vector(ratingsTriples[0: splitPoint - 1],
                                                                         ratingsTriples[splitPoint:])
            self.all_records[i] = matrix_rating
            self.input_mask[i] = input_mask
            self.output_mask[i] = output_mask
            i += 1

        for key in keys_test:
            ratingsTriples = self.userList[key]
            splitPoint = rd.randint(1, len(ratingsTriples) - 1)
            rd.shuffle(ratingsTriples)
            matrix_rating, input_mask, output_mask = self.triples2vector(ratingsTriples[0: splitPoint - 1],
                                                                         ratingsTriples[splitPoint:])
            self.all_records[i] = matrix_rating
            self.input_mask[i] = input_mask
            self.output_mask[i] = output_mask
            i += 1

    def shuffle_data(self):
        i = 0
        keys_train_dev, keys_test = list(self.userList.keys())[0: self.num_dev + self.num_train], list(self.userList.keys())[self.num_dev + self.num_train:]
        rd.shuffle(keys_train_dev)
        for key in keys_train_dev:
            ratingsTriples = self.userList[key]
            """ratings before i would be trainset, exclusive, that is [0, i-1]"""
            splitPoint = rd.randint(1, len(ratingsTriples) - 1)
            rd.shuffle(ratingsTriples)
            matrix_rating, input_mask, output_mask = self.triples2vector(ratingsTriples[0: splitPoint - 1],
                                                                         ratingsTriples[splitPoint:])
            self.all_records[i] = matrix_rating
            self.input_mask[i] = input_mask
            self.output_mask[i] = output_mask
            i += 1

    def triples2vector(self, triples_input, triples_output):
        matrix_input = np.zeros((1, self.M))
        matrix_output = np.zeros((1, self.M))

        for triple in triples_input:
            matrix_input[0, self.movieID2index[triple[0]]] = triple[1]
        input_mask = matrix_input > 0

        for triple in triples_output:
            matrix_output[0, self.movieID2index[triple[0]]] = triple[1]
        output_mask = matrix_output > 0

        if np.sum(np.ones((1, self.M))[np.multiply(input_mask, output_mask)]) > 0:
            print('There is overlapping')

        return matrix_input + matrix_output, input_mask, output_mask

    def split_set(self, ratio_train, ratio_dev):
        self.num_train = int(self.user * ratio_train)
        self.num_dev = int(self.user * ratio_dev)
        self.num_test = self.user - self.num_train - self.num_dev

        self.index_list_train = list(range(0, self.num_train))
        self.index_list_dev = list(range(self.num_train, self.num_train + self.num_dev))
        self.index_list_test = list(range(self.num_train + self.num_dev, self.user))

    def get_batch_train(self, batch_size):
        if self.used_train + batch_size >= self.num_train:
            return None, None, None, False

        ratings = np.zeros((batch_size, self.M))
        out_mask = np.ones((batch_size, self.M), dtype=bool)
        in_mask = np.ones((batch_size, self.M), dtype=bool)
        for i in range(0, batch_size):
            ratings[i] = self.all_records[self.index_list_train[i + self.used_train], :]
            out_mask[i] = self.output_mask[self.index_list_train[i + self.used_train], :]
            in_mask[i] = self.input_mask[self.index_list_train[i + self.used_train], :]
        self.used_train += batch_size

        return ratings, out_mask, in_mask, True

    def get_batch_dev(self, batch_size):
        if self.used_dev + batch_size >= self.num_dev:
            return None, None, None, False
        ratings = np.zeros((batch_size, self.M))
        out_mask = np.ones((batch_size, self.M), dtype=bool)
        in_mask = np.ones((batch_size, self.M), dtype=bool)
        for i in range(0, batch_size):
            ratings[i] = self.all_records[self.index_list_dev[i + self.used_dev], :]
            out_mask[i] = self.output_mask[self.index_list_dev[i + self.used_dev], :]
            in_mask[i] = self.input_mask[self.index_list_dev[i + self.used_dev], :]
        self.used_dev += batch_size
        return ratings, out_mask, in_mask, True

    def get_batch_test(self, batch_size):
        if self.used_test + batch_size >= self.num_test:
            return None, None, None, False
        ratings = np.zeros((batch_size, self.M))
        out_mask = np.ones((batch_size, self.M), dtype=bool)
        in_mask = np.ones((batch_size, self.M), dtype=bool)
        for i in range(0, batch_size):
            ratings[i] = self.all_records[self.index_list_test[i + self.used_test], :]
            out_mask[i] = self.output_mask[self.index_list_test[i + self.used_test], :]
            in_mask[i] = self.input_mask[self.index_list_test[i + self.used_test], :]
        self.used_test += batch_size
        return ratings, out_mask, in_mask, True

    def renew_train(self):
        self.used_train = 0
        rd.shuffle(self.index_list_train)
        self.prepare_data()

    def renew_test(self):
        self.used_test = 0

    def renew_dev(self):
        self.used_dev = 0
        rd. shuffle(self.index_list_train)


if __name__ == '__main__':
    myData = Data_user('../ml-1m/ratings.dat')
    myData.split_set(0.8, 0.1)
    myData.prepare_data()

    # index = 0
    # while True:
    #     index += 1
    #     print(index)
    #
    #     r, i_m, o_m, flag = myData.get_batch_train(512)
    #     if flag == False:
    #         break
    #
    # myData.renew_train()
    # while True:
    #     index += 1
    #     print(index)
    #
    #     r, i_m, o_m, flag = myData.get_batch_train(512)
    #     if flag == False:
    #         break

    x, input_m, output_m, flag = myData.get_batch_test(512)
    myData.renew_test()
    test_x, input_m_t, output_m_t, _ = myData.get_batch_test(512)

    print(x)
    print(x[input_m])
    print(x[output_m])
    print(test_x)
    print(input_m_t)
    print(output_m_t)

