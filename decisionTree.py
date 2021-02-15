from typing import List
import random
from progress.bar import Bar
import auxiliary as ax

class decisionTree:
    """docstring for decisionTree."""

    def __init__(self, t: List[ax.trainingExample], maxDepth: int):

        self.root = None
        self.trainingData = t #a list of training examples
        self.maxDepth = maxDepth
        self.numFeatures = len(t[0].featureVector)

    def train(self):

        featureSet = list(range(0, self.numFeatures))
        labelSet = list(set([x.label for x in self.trainingData]))
        labelIdx = 0
        data = self.trainingData
        self.root = decisionTree.train_rec(data, labelSet, featureSet, self.maxDepth)

    @classmethod
    def train_rec(self, data, labelSet_l, featureSet, depth, nodeVal=None):

        currNode = ax.treeNode()
        currNode.val = nodeVal

        c_list = [0] * len(labelSet_l)
        labelSet = dict(zip(labelSet_l, c_list))

        for x in labelSet.keys():
            if(all(i.label == x for i in data)):
                currNode.label = x
                currNode.isLeaf = True
                return currNode
            else:
                labelSet[x] = [i.label for i in data].count(x)


        currNode.label = max(labelSet, key=labelSet.get)

        if(len(featureSet) == 0 or depth == 0):
            currNode.isLeaf = True
            return currNode



        #Find the best attribute/feature from the set of attributes/features.

        currFeat = random.choice(featureSet)

        #For now we shall choose a random attribute/feature.

        currNode.decidingFeature = currFeat

        #create a set of all possible value for that attribute/feature

        currFeatVals = list(set([i.featureVector[currFeat] for i in data]))

        for z in currFeatVals:
            td = ax.filterOn(currFeat, z, data)
            nfs = featureSet.copy()
            nfs.remove(currFeat)
            x = decisionTree.train_rec(td, labelSet_l, nfs, depth-1, nodeVal=z)
            currNode.children.append(x)

        if(len(currNode.children) == 1):
            currNode.label = currNode.children[0].label
            currNode.isLeaf = True

        return currNode

    def __str__(self, level=0, node=None):

        if(self.maxDepth > 2):
            return ""

        if(level == 0):
            node = self.root

        if(node.isLeaf):
            ret = "\t"*level+'Node value = '+str(node.val)+"\n"
            ret += "\t"*level+'Label value = '+str(node.label)+"\n\n"
        else:
            ret = "\t"*level+'Node value = '+str(node.val)+"\n"
            ret+= "\t"*level+' Feature f'+str(node.decidingFeature+1)
            ret+= str(set([i.featureVector[node.decidingFeature] for i in self.trainingData]))+"\n\n"

            for child in node.children:
                ret+= self.__str__(level+1, child)

        return ret

    def testSingle(self, t: ax.trainingExample, root=None):

        if(root):
            currNode = root
        else:
            currNode = self.root

        while(currNode):

            attNo = currNode.decidingFeature
            startNode = currNode

            if(currNode.isLeaf):

                return (currNode.label, t.label, currNode.label == t.label)

            for child in currNode.children:

                if(t.featureVector[attNo] == child.val):
                    currNode = child
                    break

            if(startNode == currNode):
                break

        return (currNode.label, t.label, currNode.label == t.label)




    def testBatch(self, data):

        retList = []
        k = 0
        bar = Bar('Processing', max=len(data))

        for x in data:

            root = self.root
            retList.append(self.testSingle(t=x, root=root))
            bar.next()

        bar.finish()
        return retList
