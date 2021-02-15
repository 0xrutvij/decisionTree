import pandas as pd
import decisionTree as dt
import auxiliary as ax

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

df = pd.read_csv('./csv/monks-1.train.csv')


trainingData = []

for index, rows in df.iterrows():

    my_list = [rows.f1, rows.f2, rows.f3, rows.f4, rows.f5, rows.f6,]
    label = rows.label
    t = ax.trainingExample(label, my_list)
    trainingData.append(t)

someTree = dt.decisionTree(trainingData, 6)

someTree.train()

print(str(someTree))

#print(someTree.testSingle(lst[1]))

testResult = someTree.testBatch(trainingData)

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

print('Error percent:', (numMiss*100)/(numMiss+numHit))


'''
       PV ->
AV    _P_ _N_
 | |P[TP][FN]      2|1
 V |N[FP][TN]      3|4
'''

print(TP, '|', FN)
print(FP, '|', TN)

print('\n'*3)

df = pd.read_csv('./csv/monks-1.test.csv')


testData = []

for index, rows in df.iterrows():

    my_list = [rows.f1, rows.f2, rows.f3, rows.f4, rows.f5, rows.f6,]
    label = rows.label
    t = ax.trainingExample(label, my_list)
    testData.append(t)

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

print('Error percent (test data):', (numMiss*100)/(numMiss+numHit))


'''
       PV ->
AV    _P_ _N_
 | |P[TP][FN]      2|1
 V |N[FP][TN]      3|4
'''

print(TP, '|', FN)
print(FP, '|', TN)
