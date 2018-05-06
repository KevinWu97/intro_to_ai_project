import random
import os

from feature import getLabels, getAllFeatures, getFeatures
from naiveBayes import naiveBayes

#percentSample passed as integer value between 1 and 100
def testBayes(numClass, percentSample):
	#select for sample
	trainFilePath = os.path.join(os.getcwd(), "digitdata_train_split")
	fullLabels = getLabels(trainFilePath)
	popList = []
	for i in range(len(fullLabels)):
		popList.append(i)
	sampleSize = (int)(len(fullLabels) * percentSample / 100)
	sampleList = random.sample(popList, sampleSize)
	trainLabels = []
	for i in range(len(sampleList)):
		trainLabels.append(fullLabels[sampleList[i]])

	#retrieve image files as features
	trainFeatureList = getFeatures(trainFilePath, "number", sampleList)

	#initialize Bayes object and train using features
	bayes = naiveBayes()
	for i in range(len(trainLabels)):
		trainFeatureList.insert(0,trainLabels)
	bayes.train(numClass, trainFeatureList)

	#run test cases and return percent accuracy
	testFilePath = os.path.join(os.getcwd(), "digitdata_test_split")
	testLabels = getLabels(testFilePath)
	testFeatureList = getAllFeatures(testFilePath, "digit", len(testLabels))
	testResults = bayes.test(testFeatureList)
	successCount = 0
	for i in range(len(testLabels)):
		if(testLabels[i] == testResults[i]):
			successCount += 1
	successRate = successCount/len(testLabels)
	return successRate

print(testBayes(10, 10))