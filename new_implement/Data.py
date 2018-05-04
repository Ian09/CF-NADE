"""
Author: Dynsk

Use case:
    myData = Data('../ml-1m/ratings.dat')
    myData.split_sets({'train': 0.9, 'test': 0.1})
    myData.get_batch_new_2(512, 'train')  # (batchSize * numberMovie * 2, batchSize * 2)

    When run out of dataset, myData.get_batch_new_2() returns False

    If you want to begin a new epoch, myData.renew() and you could do the same thing as before
"""

import random as rd
import numpy as np

class Data:

    def __init__(self, data_directory):
        self.userList = {}
        self.movieID2index = {}
        index = 0
        self.sampleList = []
        self.splitDict = {}
        self.used_index = 0  # indicator for how much data is used
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

        for key in self.userList:
            print('processing user:' + key)
            ratingsTriples = self.userList[key]
            """ratings before i would be trainset, exclusive, that is [0, i-1]"""
            splitPoint = rd.randint(1, len(ratingsTriples) - 1)
            ratingsTriplesSorted = sorted(ratingsTriples, key=lambda x: x[2])
            for triple in ratingsTriplesSorted[splitPoint:]:
                self.sampleList.append((ratingsTriplesSorted[0: splitPoint - 1], triple))


    def split_sets(self, name_portion_dict):
        rd.shuffle(self.sampleList)
        currentPos = 0
        for key in name_portion_dict:
            newPos = int(name_portion_dict[key] * self.sampleList.__len__() + currentPos)
            self.splitDict[key] = self.sampleList[currentPos: newPos]
            currentPos = newPos


    def triples2vector(self, triples):
        """transform (movieID, ratings, timStamp) triples to 1 * dim * 2 vector"""
        outputVector = np.zeros((1, self.movieDim, 2))
        for triple in triples:
            outputVector[0, self.movieID2index[triple[0]], 0] = triple[1]  # adding rating
            outputVector[0, self.movieID2index[triple[0]], 1] = triple[2]  # adding timeStamp
        return outputVector

    def dense2sparseVector(self, sampleSet):
        """
        :param sampleSet: a set contains two elements. The first one is a list of all triples of input, the other
        is a triple of the corresponding output
        :return:
        """
        inputVector = self.triples2vector(sampleSet[0])
        outputVector = self.triples2vector((sampleSet[1]))
        finalVector = np.zeros((1, self.movieDim, 2, 2))
        finalVector[:, :, :, 0] = inputVector
        finalVector[:, :, :, 1] = outputVector
        return finalVector

    def dense2sparseVector_new(self, sampleSet):
        """
        :param sampleSet: a set contains two elements. The first one is a list of all triples of input, the other
        is a triple of the corresponding output
        :return:
        """
        inputVector = self.triples2vector(sampleSet[0])
        outputVector = np.zeros((1, 2), dtype=int)

        movieID, rating, timeStamp = sampleSet[1]  # unpack output triple
        inputVector[0, :, 1] = timeStamp - inputVector[0, :, 1]  # timeStamp difference
        outputVector[0, 0] = rating
        outputVector[0, 1] = self.movieID2index[movieID]
        return (inputVector, outputVector)




    def get_batch(self, batchSize, setName):
        """
        Aborted
        :param batchSize:
        :param setName:
        :return:
        """
        if len(self.splitDict[setName]) < batchSize:
            return False
        outputVector = np.zeros((batchSize, self.movieDim, 2, 2))
        for i in range(0, batchSize):
            try:
                sample = self.splitDict[setName].pop(0)
                outputVector[i,:,:,:] = self.dense2sparseVector(sample)
            except:
                i -= 1
        return outputVector

    def get_batch_new(self, batchSize, setName):
        """
        Aborted
        :param batchSize: usually 512
        :param setName: 'train' or 'test'
        :return: a tuple, containing batchSize * movie_dim * 2 matrix and a batchSize * 2 matrix

        batchSize * movie_dim * 2: first dimension is for ratings and the second is for timestamp difference
        batchSize * 2 matrix: first column is for ratings and the second column is for index of the movie
        """
        if len(self.splitDict[setName]) < batchSize:
            return False
        outputMatrix_input = np.zeros((batchSize, self.movieDim, 2), dtype=int)
        outputMatrix_output = np.zeros((batchSize, 2), dtype=int)
        for i in range(0, batchSize):
            try:
                sample = self.splitDict[setName].pop(0)
                outputMatrix_input[i, :, :], outputMatrix_output[i, :] = self.dense2sparseVector_new(sample)
            except:
                i -= 1
        print(outputMatrix_output)
        return (outputMatrix_input, outputMatrix_output)

    def get_batch_new_2(self, batchSize, setName):
        """
        Could use self.renew() to begin a new epoch

        :param batchSize: usually 512
        :param setName: 'train' or 'test'
        :return: a tuple, containing batchSize * movie_dim * 2 matrix and a batchSize * 2 matrix

        batchSize * movie_dim * 2: first dimension is for ratings and the second is for timestamp difference
        batchSize * 2 matrix: first column is for ratings and the second column is for index of the movie
        """
        if len(self.splitDict[setName]) - self.used_index < batchSize:
            return False
        outputMatrix_input = np.zeros((batchSize, self.movieDim, 2), dtype=int)
        outputMatrix_output = np.zeros((batchSize, 2), dtype=int)
        for i in range(0, batchSize):
            sample = self.splitDict[setName][self.used_index + i]
            outputMatrix_input[i, :, :], outputMatrix_output[i, :] = self.dense2sparseVector_new(sample)
        self.used_index += batchSize
        return outputMatrix_input, outputMatrix_output

    def renew(self):
        self.used_index = 0


if __name__ == '__main__':
    myData = Data('../ml-1m/ratings.dat')
    myData.split_sets({'train': 0.9, 'test': 0.1})
    index = 0

    while True:
        flag = myData.get_batch_new_2(512, 'train')
        if flag == False:
            break
        index += 1
        print('batch: %d' % index)

    myData.renew()

    print('begin a new epoch')
    while True:
        flag = myData.get_batch_new_2(512, 'train')
        if flag == False:
            break
        index += 1
        print('batch: %d' % index)
