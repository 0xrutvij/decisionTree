import pandas as pd
import decisionTree as dt
import auxiliary as ax

# function to create the training data
def createTrainingData(trainingData):
    for index, rows in df.iterrows():

        my_list = [rows.f1, rows.f2, rows.f3, rows.f4, rows.f5, rows.f6,]
        label = rows.label
        t = ax.trainingExample(label, my_list)
        trainingData.append(t)
        
    return trainingData

# function to create the test data
def createTestData(testData):
    
    for index, rows in df.iterrows():
        my_list = [rows.f1, rows.f2, rows.f3, rows.f4, rows.f5, rows.f6,]
        label = rows.label
        t = ax.trainingExample(label, my_list)
        testData.append(t)
    
    return testData 

def findPosNegValues(testResult):
    
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

    print('Training Error Percent:', (numMiss*100)/(numMiss+numHit))


    '''
        PV ->
    AV    _P_ _N_
    | |P[TP][FN]      2|1
    V |N[FP][TN]      3|4
    '''

    print(TP, '|', FN)
    print(FP, '|', TN)

    print('\n'*3)

    testResult2 = someTree.testBatch(testData)

    numMiss = 0
    numHit = 0
    FN = TN = TP = FP = 0

    for x in testResult2:

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

    print('Test Error Percent:', (numMiss*100)/(numMiss+numHit))


    '''
        PV ->
    AV    _P_ _N_
    | |P[TP][FN]      2|1
     V |N[FP][TN]      3|4
    '''

    print(TP, '|', FN)
    print(FP, '|', TN)
    

# function to create the decision tree and find the training and test errors
def trainAndTest(trainingData, x, testData):

    someTree = dt.decisionTree(trainingData, x)

    someTree.train()

    print(str(someTree))

    testResult = someTree.testBatch(trainingData)

    
