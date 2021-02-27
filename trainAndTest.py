import pandas as pd
import decisionTree as dt
import auxiliary as ax

# function to create the training data
def createTrainingData(trainingData, df):
    for index, rows in df.iterrows():

        my_list = [rows.f1, rows.f2, rows.f3, rows.f4, rows.f5, rows.f6,]
        label = rows.label
        t = ax.trainingExample(label, my_list)
        trainingData.append(t)
        
    return trainingData

# function to create the test data
def createTestData(testData, df):
    
    for index, rows in df.iterrows():
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
    print('Error: ',error)

    # If makeConfusion = 1, make a confusion matrix
    if makeConfusion==1:
        '''
            PV ->
        AV    _P_ _N_
        | |P[TP][FN]      2|1
        V |N[FP][TN]      3|4
        '''

        print(TP, '|', FN)
        print(FP, '|', TN)

    return error

# function to create the decision tree and find the training and test errors
def trainAndTest(trainingData, x, testData, makeConfusion):

    # create the decision tree
    someTree = dt.decisionTree(trainingData, x)
    
    # train the tree
    someTree.train()

    print(str(someTree))

    # find training error and make confusion matrix if needed
    testResult = someTree.testBatch(trainingData)
    print('Results for Training Data: ')
    trainingError = findResults(testResult, makeConfusion)
    
    print('\n'*3)
    
    # find test error and make confusion matrix if needed
    testResult2 = someTree.testBatch(testData)
    print('Results for Test Data: ')
    testError = findResults(testResult2, makeConfusion)
    
    List = [trainingError, testError] 
    
    return List
