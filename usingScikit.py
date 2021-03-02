from sklearn import tree
import graphviz

def CreateScikitTree(trainingData):
  
  # Set the labels and features
  labels_Y = [trainingExample.label for trainingExample in trainingData]
  features_X = [trainingExample.featureVector for trainingExample in trainingData]

  dtClassifier = tree.DecisionTreeClassifier()
  dtClassifier = dtClassifier.fit(features_X, labels_Y)

  # Find the training error
  preds = dtClassifier.predict(features_X)
  
  ## Check training errors
  trainErrors = []
  #print(preds)

  for pred, Y in zip(preds, labels_Y):
    trainErrors.append((pred, Y, pred==Y))

  ## Check test errors
  test_labels_Y = [trainingExample.label for trainingExample in testData]
  test_features_X = [trainingExample.featureVector for trainingExample in testData]
  preds = dtClassifier.predict(test_features_X)
  #print(preds)

  testErrors = []
  for pred, Y in zip(preds, test_labels_Y):
    testErrors.append((pred, Y, pred==Y))

  dot_data = tree.export_graphviz(dtClassifier, out_file=None)
  graph = graphviz.Source(dot_data)
  graph.render("./plots/sciSPECT", format="png")
  
  return testErrors
