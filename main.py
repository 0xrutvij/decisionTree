import pandas as pd
import decisionTree as dt
import auxiliary as ax
import trainAndTest as tandt

'''
lst = []

lst.append(trainingExample(1, [1, 2, 3]))
lst.append(trainingExample(1, [1, 3, 2]))
lst.append(trainingExample(1, [2, 3, 1]))
lst.append(trainingExample(1, [2, 1, 3]))
lst.append(trainingExample(0, [3, 1, 2]))
lst.append(trainingExample(0, [3, 2, 1]))
lst.append(trainingExample(0, [1, 2, 3]))
lst.append(trainingExample(0, [1, 2, 3]))
lst.append(trainingExample(0, [1, 2, 3]))
lst.append(trainingExample(0, [1, 2, 3]))
'''

## Monks-1

# create the training data for monks-1
df = pd.read_csv('./csv/monks-1.train.csv')
trainingData = []
trainingData = tandt.createTrainingData(trainingData, df)

# create the test data for monks-1
df = pd.read_csv('./csv/monks-1.test.csv')
testData = []
testData = tandt.createTestData(testData, df)

trainingErrorList1 = []
testErrorList1 = []

# create trees of depth 1 and 2 for monks-1
for x in range(1,3):
    # create the trees of various depths
    errors = tandt.trainAndTest(trainingData, x, testData, 1) # create confusion matrix
    trainingErrorList1.append(errors[0])
    testErrorList1.append(errors[1])
    
# for depth = 3,...,10 create tree for monks-1
for x in range(3,11):
    # create the trees of various depths
    errors = tandt.trainAndTest(trainingData, x, testData, 0)
    trainingErrorList1.append(errors[0])
    testErrorList1.append(errors[1])

## Monks-2

# create the training data for monks-2
df = pd.read_csv('./csv/monks-2.train.csv')
trainingData = []
trainingData = tandt.createTrainingData(trainingData, df)

# create the test data for monks-2
df = pd.read_csv('./csv/monks-2.test.csv')
testData = []
testData = tandt.createTestData(testData, df)

trainingErrorList2 = []
testErrorList2 = []

# for depth = 1,...,10 create tree for monks-2
for x in range(1,11):
    # create the trees of various depths
    errors = tandt.trainAndTest(trainingData, x, testData, 0)
    trainingErrorList2.append(errors[0])
    testErrorList2.append(errors[1])

## Monks-3

# create the training data for monks-3
df = pd.read_csv('./csv/monks-3.train.csv')
trainingData = []
trainingData = tandt.createTrainingData(trainingData, df)

# create the test data for monks-3
df = pd.read_csv('./csv/monks-3.test.csv')
testData = []
testData = tandt.createTestData(testData, df)

trainingErrorList3 = []
testErrorList3 = []

# for depth = 1,...,10 create tree for monks-3
for x in range(1,11):
    # create the trees of various depths
    errors = tandt.trainAndTest(trainingData, x, testData, 0)
    trainingErrorList3.append(errors[0])
    testErrorList3.append(errors[1])
