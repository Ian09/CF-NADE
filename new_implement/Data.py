"""
Author: Dynsk

Use case:
    myData = Data('../ml-1m/ratings.dat')
    myData.split_sets({'train': 0.9, 'test': 0.1})
    myData.get_batch_new(512, 'train')  # (batchSize * numberMovie * 2, batchSize * 2)

    When run out of dataset, myData.get_batch() returns False
"""

import random as rd
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataDir', type=str, default='../ml-1m/ratings.dat')
parser.add_argument('--train', type=float, default=0.9)
parser.add_argument('--test', type=float, default=0.1)
parser.add_argument('--saveDir', type=str, default='./data')

args = parser.parse_args()

class Data:

    def __init__(self, data_directory):
        self.userList = {}
        self.movieID2index = {}
        index = 0
        self.sampleList = []
        self.splitDict = {}
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
            ratingsTriples = self.userList[key]
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

    def prepareData(self, saveDir):
        self.train_X = np.zeros((len(self.splitDict['train']), self.movieDim, 2))
        self.train_Y = np.zeros((len(self.splitDict['train']), 2))
        self.test_X = np.zeros((len(self.splitDict['test']), self.movieDim, 2))
        self.test_Y = np.zeros((len(self.splitDict['test']), 2))
        print(len(self.splitDict['test']))
        for i in range(len(self.splitDict['train'])):
            self.train_X[i,:,:], self.train_Y[i,:] = self.dense2sparseVector_new(self.splitDict['train'][i])
        for i in range(len(self.splitDict['test'])):
            self.test_X[i,:,:], self.test_Y[i,:] = self.dense2sparseVector_new(self.splitDict['test'][i])
        np.savetxt(os.path.join(saveDir, 'train_X.dat'), self.train_X, delimiter=',')
        np.savetxt(os.path.join(saveDir, 'train_Y.dat'), self.train_Y, delimiter=',')
        np.savetxt(os.path.join(saveDir, 'test_X.dat'), self.test_X, delimiter=',')
        np.savetxt(os.path.join(saveDir, 'test_Y.dat'), self.test_Y, delimiter=',')


    def get_batch_new(self, batchSize, setName):
        """
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
        #print(outputMatrix_output)
        return (outputMatrix_input, outputMatrix_output)





if __name__ == '__main__':
    myData = Data(args.dataDir)
    myData.split_sets({'train': args.train, 'test': args.test})
    if not os.path.exists(args.saveDir):
        os.makedirs(args.saveDir)
    myData.prepareData(args.saveDir)


