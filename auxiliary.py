class trainingExample:

    def __init__(self, label, featureVector):
        self.label = label
        #a List containing all the feature vals
        self.featureVector = featureVector

class treeNode:

    def __init__(self):
        self.val = None
        self.label = None
        self.isLeaf = False
        self.decidingFeature = None
        self.children = []


def filterOn(attNo, attVal, td):
    retList = []
    for i in td:
        if(i.featureVector[attNo] == attVal):
            retList.append(i)
        else:
            continue
    return retList
