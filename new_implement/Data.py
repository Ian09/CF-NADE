"""
Author: Dynsk

Use case:
    myData = Data('../ml-1m/ratings.dat')
    myData.split_sets({'train': 0.9, 'test': 0.1})
    myData.get_batch(512, 'train')  # batchSize * numberMovie * 2 * 2

    When run out of dataset, myData.get_batch() returns False
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
        with open(data_directory, 'r') as f:
            for line in f.readlines():
                userID, movieID, rating, timeStamp = line.split('::')
                timeStamp = timeStamp.strip()
                if int(movieID) not in self.movieID2index:
                    self.movieID2index[int(movieID)] = index
                    index += 1

                if userID not in self.userList:
                    self.userList[userID] = [(int(movieID), int(rating), int(timeStamp))]
                else:
                    self.userList[userID].append((int(movieID), int(rating), int(timeStamp)))

        self.movieDim = len(self.movieID2index)

        for key in self.userList:
            print('processing user:' + key)
            ratingsTriples = self.userList[key]
            """ratings before i would be trainset, exclusive, that is [0, i-1]"""
            splitPoint = rd.randint(1, len(ratingsTriples) - 1)
            ratingsTriplesSorted = sorted(ratingsTriples, key=lambda x: x[2])
            # inputVector = self.triples2vector(ratingsTriplesSorted[0: splitPoint - 1])
            for triple in ratingsTriplesSorted:
                # outputVector = self.triples2vector([triple])
                # finalVector = np.zeros((1, self.movieDim, 2, 2))
                # finalVector[:, :, :, 0] = inputVector
                # finalVector[:, :, :, 1] = outputVector
                # self.sampleList.append(finalVector)
                self.sampleList.append((ratingsTriplesSorted[0 : splitPoint - 1], triple))


    def split_sets(self, name_portion_dict):
        rd.shuffle(self.sampleList)
        currentPos = 0
        for key in name_portion_dict:
            newPos = int(name_portion_dict[key] * self.sampleList.__len__() + currentPos)
            self.splitDict[key] = self.sampleList[currentPos : newPos]
            currentPos = newPos

        rd.shuffle(self.sampleList)

    def triples2vector(self, triples):
        """transform (movieID, ratings, timStamp) triples to 1 * dim * 2 vector"""
        outputVector = np.zeros((1, self.movieDim, 2))
        for triple in triples:

            outputVector[0, self.movieID2index[triple[0]], 0] = triple[1]

            outputVector[0, self.movieID2index[triple[0]], 1] = triple[2]


        return outputVector

    def dense2sparseVector(self, sampleSet):
        inputVector = self.triples2vector(sampleSet[0])
        outputVector = self.triples2vector((sampleSet[1]))
        finalVector = np.zeros((1, self.movieDim, 2, 2))
        finalVector[:, :, :, 0] = inputVector
        finalVector[:, :, :, 1] = outputVector
        return finalVector

    def get_batch(self, batchSize, setName):
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





if __name__ == '__main__':
    myData = Data('../ml-1m/ratings.dat')
    myData.split_sets({'train': 0.9, 'test': 0.1})
    print(myData.get_batch(512, 'train'))
    print(myData.get_batch(512, 'train'))
    print(myData.get_batch(512, 'train'))

    print('Hello world!')