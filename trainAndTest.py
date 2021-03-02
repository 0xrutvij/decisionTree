import pandas as pd
import decisionTree as dt
import auxiliary as ax
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

# function to create the training data
def createTrainingData(trainingData, df, SPECTmode=False):
    for index, rows in df.iterrows():
        if SPECTmode:
            my_list = [rows.a,rows.b,rows.c,rows.d,rows.e,
            rows.f,rows.g,rows.h,rows.i,rows.j,rows.k,rows.l,
            rows.m,rows.n,rows.o,rows.p,rows.q,rows.r,rows.s,rows.t,rows.u,rows.v]
        else:
            my_list = [rows.f1, rows.f2, rows.f3, rows.f4, rows.f5, rows.f6,]
        label = rows.label
        t = ax.trainingExample(label, my_list)
        trainingData.append(t)

    return trainingData

# function to create the test data
def createTestData(testData, df, SPECTmode=False):

    for index, rows in df.iterrows():
        if(SPECTmode):
            my_list = [rows.a,rows.b,rows.c,rows.d,rows.e,
            rows.f,rows.g,rows.h,rows.i,rows.j,rows.k,rows.l,
            rows.m,rows.n,rows.o,rows.p,rows.q,rows.r,rows.s,rows.t,rows.u,rows.v]
        else:
            my_list = [rows.f1, rows.f2, rows.f3, rows.f4, rows.f5, rows.f6,]
        label = rows.label
        t = ax.trainingExample(label, my_list)
        testData.append(t)

    return testData

# function to compute the training and test results. If makeConfusion = 1, then
# a confusion matrix will be produced
def findResults(testResult, makeConfusion):

    numMiss = 0
    numHit = 0
    FN = TN = TP = FP = 0

    for x in testResult:

        if(x[2]):
            numHit+=1
            if(x[0]):
                TP+=1
            else:
                TN+=1
        else:
            numMiss+=1
            if(x[0]):
                FP+=1
            else:
                FN+=1

    error = (numMiss*100)/(numMiss+numHit)

    # If makeConfusion = 1, make a confusion matrix
    if makeConfusion==1:
        '''
            PV ->
        AV    _P_ _N_
        | |P[TP][FN]      2|1
        V |N[FP][TN]      3|4
        '''

        print(str(TP).rjust(3, '0'), '|', str(FN).rjust(3, '0'))
        print(str(FP).rjust(3, '0'), '|', str(TN).rjust(3, '0'))

    return error

# function to create the decision tree and find the training and test errors
# makeReport = 1 indicates that the tree needs to be printed and
# a confusion matrix should be produced
def trainAndTest(trainingData, x, testData, makeReport):

    # create the decision tree
    someTree = dt.decisionTree(trainingData, x)

    # train the tree
    someTree.train()

    if makeReport == 1:
        print('Decision Tree of Depth', x)
        print(str(someTree))

    # find training error and make confusion matrix if needed
    testResult = someTree.testBatch(trainingData)
    if makeReport ==1:
        print('The Confusion Maxtrix on the Training Set for Depth of ', x, ':')
    trainingError = findResults(testResult, makeReport)

    # find test error and make confusion matrix if needed
    testResult2 = someTree.testBatch(testData)
    if makeReport ==1:
        print('The Confusion Maxtrix on the Test Set for Depth of ', x, ':')
    testError = findResults(testResult2, makeReport)

    List = [trainingError, testError]

    return List

# plotting training and testing error curves together
# with tree depth on the x-axis and error on the y-axis
def plotErrors(trainingErrorList, testErrorList, monkNum, SPECTmode=False):
    plt.plot(np.arange(1,11,1),trainingErrorList, label = 'Training Error')
    plt.plot(np.arange(1,11,1),testErrorList, label = 'Test Error')
    plt.xlabel('Depth')
    plt.ylabel('Errors')
    if SPECTmode:
        title = 'Training and Testing Error Curves for SPECT data'
        save_file = 'SPECT.png'
    else:
        title = 'Training and Testing Error Curves for Monk' + monkNum
        save_file = 'Monk' + monkNum + '.png'
    plt.title(title)
    plt.legend()
    plt.savefig('plots/'+save_file, bbox_inches='tight')
    plt.show()

# function to display the table of training and test errors per depth
def displayTable(trainingErrorList,testErrorList):
    trainingErrorList.insert(0, 'Training Error')
    testErrorList.insert(0, 'Test Error')
    print(tabulate([['Depth',1,2,3,4,5,6,7,8,9,10], trainingErrorList, testErrorList]))
