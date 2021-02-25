class trainingExample:
    """docstring for trainingExample. A class to hold
    a single training Example/test example. i.e. one row
    from the data file """

    def __init__(self, label, featureVector):
        #constructor for the trainingExample class
        #class label of the example
        self.label = label
        #a List containing all the feature values
        self.featureVector = featureVector

class treeNode:
    """docstring for treeNode. A class representing a single node of the tree"""

    def __init__(self):
        #constructor for the treeNode class
        #Value of the node, in our context the value to which
        #we must match ancestor node's deciding feature.
        self.val = None
        #If this the node at which we stop looking, what label will we assign to
        #the example in question
        self.label = None
        #A boolean representing whether or not the node is a leaf node
        self.isLeaf = False
        #If the node isn't a leaf, the feature on which we will split the children
        self.decidingFeature = None
        #List of children node for this node
        self.children = []


def filterOn(attNo, attVal, td):
    """An helper function to filter the training data by the current nodes
    decidingFeature, the data passed to the children is the data whose value matches
    the value of the child node's, where the decidingFeature is defined by the parent"""
    retList = []
    for i in td:
        if(i.featureVector[attNo] == attVal):
            retList.append(i)
        else:
            continue
    return retList
