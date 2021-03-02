import pandas as pd
import decisionTree as dt
import auxiliary as ax
import trainAndTest as tandt

for monkNum in range (1, 4):
    df = pd.read_csv('./csv/monks-'+str(monkNum)+'.train.csv')
    trainingData = []
    trainingData = tandt.createTrainingData(trainingData, df)

    # create the test data for monks-1to3
    df = pd.read_csv('./csv/monks-'+str(monkNum)+'.test.csv')
    testData = []
    testData = tandt.createTestData(testData, df)

    trainingErrorList1 = []
    testErrorList1 = []

    # create trees of depth 1 and 2 for monks-1to3
    print('MONK Problem '+str(monkNum)+': ')
    for x in range(1,3):
        # create the trees of various depths
        errors = tandt.trainAndTest(trainingData, x, testData, 1) # create confusion matrix and print tree
        trainingErrorList1.append(errors[0])
        testErrorList1.append(errors[1])

    # for depth = 3,...,10 create tree for monks-1to3
    for x in range(3,11):
        # create the trees of various depths
        errors = tandt.trainAndTest(trainingData, x, testData, 0)
        trainingErrorList1.append(errors[0])
        testErrorList1.append(errors[1])

    # plot plotting training and testing error curves together
    # with tree depth on the x-axis and error on the y-axis
    tandt.plotErrors(trainingErrorList1, testErrorList1, str(monkNum))

    # Display the Training and Testing Errors
    tandt.displayTable(trainingErrorList1,testErrorList1)




## Spect Data

# create the training data for SPECT
df = pd.read_csv('./spectCSVs/SPECT.train.csv')
trainingData = []
trainingData = tandt.createTrainingData(trainingData, df, SPECTmode=True)

# create the test data for SPECT
df = pd.read_csv('./spectCSVs/SPECT.test.csv')
testData = []
testData = tandt.createTestData(testData, df, SPECTmode=True)

trainingErrorList1 = []
testErrorList1 = []

# create trees of depth 1 and 2 for SPECT
print('SPECT Data: ')
for x in range(1,3):
    # create the trees of various depths
    errors = tandt.trainAndTest(trainingData, x, testData, 1) # create confusion matrix and print tree
    trainingErrorList1.append(errors[0])
    testErrorList1.append(errors[1])

# for depth = 3,...,10 create tree for SPECT
for x in range(3,11):
    # create the trees of various depths
    errors = tandt.trainAndTest(trainingData, x, testData, 0)
    trainingErrorList1.append(errors[0])
    testErrorList1.append(errors[1])

# plot plotting training and testing error curves together
# with tree depth on the x-axis and error on the y-axis
tandt.plotErrors(trainingErrorList1, testErrorList1, '1', SPECTmode=True)

# Display the Training and Testing Errors
tandt.displayTable(trainingErrorList1,testErrorList1)
