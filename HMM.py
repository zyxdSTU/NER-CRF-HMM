from numpy import *
import numpy
import numpy as np
import operator
from util import *
from seqeval.metrics import classification_report

class HMM():
    '''
    wordDcitSize 词表长度
    tagDictSize 标签字典长度
    emitProb 观测概率矩阵
    transitionProb 转移概率矩阵
    initProb 初始概率向量
    '''
    def __init__(self, wordDictSize, tagDictSize):
        self.wordDictSize = wordDictSize
        self.tagDictSize = tagDictSize

        #初始以及归一化 转移概率矩阵、发射概率矩阵、初始概率矩阵
        self.transitionProb = np.random.rand(self.tagDictSize, self.tagDictSize)
        for index in range(self.tagDictSize):
            self.transitionProb[index] = self.transitionProb[index] / np.sum(self.transitionProb[index])

        self.initProb = numpy.random.rand(self.tagDictSize)
        self.initProb = self.initProb / np.sum(self.initProb)

        self.emitProb = numpy.random.rand(self.tagDictSize, self.wordDictSize)
        for index in range(self.tagDictSize):
            self.emitProb[index] = self.emitProb[index] / np.sum(self.emitProb[index])


    '''
    监督学习, 极大似然估计
    '''
    def trainSup(self, trainWordLists, trainTagLists):
        self.transitionProb = numpy.zeros((self.tagDictSize, self.tagDictSize))
        self.initProb = numpy.zeros(self.tagDictSize) 
        self.emitProb = numpy.zeros((self.tagDictSize, self.wordDictSize))

        for i in range(len(trainWordLists)):
            for j in range(len(trainWordLists[i])):
                word, tag = trainWordLists[i][j], trainTagLists[i][j]
                self.initProb[tag] += 1
                self.emitProb[tag][word] += 1
                if j < len(trainWordLists[i])-1:
                    nextTag = trainTagLists[i][j+1]
                    self.transitionProb[tag][nextTag] += 1
        self.initProb = self.initProb / (self.initProb.sum())
        for index, value in enumerate(self.emitProb.sum(axis=1)):
            if value == 0: continue
            self.emitProb[index, :] = self.emitProb[index, :] / value

        for index, value in enumerate(self.transitionProb.sum(axis=1)):
            if value == 0: continue
            self.transitionProb[index, :] = self.transitionProb[index, :] / value

        self.initProb[self.initProb == 0] = 1e-10
        self.transitionProb[self.transitionProb == 0] = 1e-10
        self.emitProb[self.emitProb == 0] = 1e-10

    '''
    前向算法
    '''
    def forwardAlg(self, sentence):
        sentenceSize = len(sentence)
        alpha = numpy.zeros((sentenceSize, self.tagDictSize))
        alpha[0] = self.initProb * self.emitProb[:,int(sentence[0])]
        for index, word in enumerate(sentence):
            if index == 0: continue
            alpha[index] = numpy.dot(alpha[index-1], self.transitionProb) * self.emitProb[:,word]
        return alpha

    '''
    后向算法
    '''
    def backwardAlg(self, sentence):
        sentenceSize = len(sentence)
        beta = numpy.zeros((sentenceSize, self.tagDictSize))
        beta[sentenceSize-1] = numpy.ones(self.tagDictSize)
        for index in reversed(range(sentenceSize)):
            if index == sentenceSize-1: continue
            beta[index] = numpy.dot(beta[index+1] * self.emitProb[:,sentence[index+1]], self.transitionProb.T)
        return beta
 
    '''
    维特比算法
    '''
    def viterbiAlg(self, sentence):
        sentenceSize = len(sentence)
        score = numpy.zeros((sentenceSize, self.tagDictSize))
        path = numpy.zeros((sentenceSize, self.tagDictSize))

        score[0] = self.initProb + self.emitProb[:,sentence[0]]

        state = numpy.zeros(sentenceSize)

        for index, word in enumerate(sentence):
            if index == 0: continue
            temp = score[index-1] + self.transitionProb.T
            path[index] = numpy.argmax(temp, axis=1)
            score[index] = [element[int(path[index,i])] for i, element in enumerate(temp)] + self.emitProb[:,word]

        state[-1] = numpy.argmax(score[-1])
        
        for i in reversed(range(sentenceSize)):
            if i == sentenceSize -1: continue
            state[i] = path[i+1][int(state[i+1])]
        return state

    def test(self, testWordLists, testTagLists, wordDict, tagDict):
        #防止溢出，对参数矩阵取对数
        self.transitionProb = numpy.log10(self.transitionProb)
        self.emitProb = numpy.log10(self.emitProb)
        self.initProb = numpy.log10(self.initProb)

        real, predict = [], []

        for sentence, tag in zip(testWordLists, testTagLists):
            tagPre = self.viterbiAlg(sentence)

            real.append(int2str(tag, tagDict))
            predict.append(int2str(tagPre, tagDict))

        print(classification_report(real, predict, digits=6))


   

