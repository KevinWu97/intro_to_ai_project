import random
import os

from feature import getLabels, getAllFeatures, getFeatures
from naiveBayes import naiveBayes

#percentSample passed as integer value between 1 and 100
def testBayes(imageType, trainFilePath, testFilePath, numClass, percentSample, smoothing):
	#select for sample
	fullLabels = getLabels(trainFilePath)
	popList = []
	for i in range(len(fullLabels)):
		popList.append(i)
	sampleSize = 0
	sampleList = []
	if percentSample < 100:
		sampleSize = (int)(len(fullLabels) * percentSample / 100)
		sampleList = random.sample(popList, sampleSize)
	else:
		sampleSize = len(fullLabels)
		for i in range(len(fullLabels)):
			sampleList.append(i)
	trainLabels = []
	for i in range(len(sampleList)):
		trainLabels.append(fullLabels[sampleList[i]])

	#retrieve image files as features
	trainFeatureList = getFeatures(trainFilePath, imageType, sampleList)

	#initialize Bayes object and train using features
	bayes = naiveBayes()
	bayes.setSmooth(smoothing)
	bayes.train(numClass, trainFeatureList, trainLabels)

	#run test cases and return percent accuracy
	testLabels = getLabels(testFilePath)
	testFeatureList = getAllFeatures(testFilePath, imageType, len(testLabels))
	testResults = bayes.test(testFeatureList)
	successCount = 0
	for i in range(len(testLabels)):
		if(testLabels[i] == testResults[i]):
			successCount += 1
	successRate = successCount/len(testLabels)
	return successRate

def optimizeSmoothing(imageType, trainFilePath, testFilePath, numClass, percentSample, lowerBound, upperBound):
	smoothingTested = []
	smoothingSuccess = []
	for i in range(lowerBound, upperBound + 1):
		smoothingTested.append(i)
		smoothingResults = testBayes(imageType, trainFilePath, testFilePath, numClass, percentSample, i)
		smoothingSuccess.append(smoothingResults)
	print(smoothingTested)
	print(smoothingSuccess)
	success = smoothingTested[0]
	percent = 0
	for i in range(len(smoothingTested)):
		if percent < smoothingSuccess[i]:
			success = smoothingTested[i]
			percent = smoothingSuccess[i]
	print("Optimal smoothing at " + str(success) + " with success rate " + str(percent))

# trainFilePath = os.path.join(os.getcwd(), "digitdata_train_split")
# testFilePath = os.path.join(os.getcwd(), "digitdata_test_split")
# prefix = "digit"
# print(testBayes(trainFilePath, testFilePath, prefix, 10, 100, 9))
# trainFilePath = os.path.join(os.getcwd(), "facedata_train_split")
# testFilePath = os.path.join(os.getcwd(), "facedata_test_split")
# prefix = "face"
# print(testBayes(trainFilePath, testFilePath, prefix, 2, 100, 1))

# trainFilePath = os.path.join(os.getcwd(), "digitdata_train_split")
# testFilePath = os.path.join(os.getcwd(), "digitdata_test_split")
# prefix = "digit"
# optimizeSmoothing(trainFilePath, testFilePath, prefix, 10, 100, 1, 10)
trainFilePath = os.path.join(os.getcwd(), "facedata_train_split")
testFilePath = os.path.join(os.getcwd(), "facedata_test_split")
prefix = "face"
optimizeSmoothing(trainFilePath, testFilePath, prefix, 2, 100, 1, 10)