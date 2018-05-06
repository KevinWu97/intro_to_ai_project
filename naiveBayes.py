class naiveBayes(object):
	def __init__(self):
		self.classProb = []
		self.featureProb = []
		self.smoothing = 1

	def train(numClass, sampleSet):
		#get count of features
		featureCount = len(sampleSet[0] - 1)

		#create matrix featureProb[classification][feature number], initialized to smoothing for all values
		self.featureProb = []
		for i in range(numClass):
			temp = []
			for i in range(featureCount):
				temp.append(smoothing)
			self.featureProb.append(temp)

		#create matrix self.classProb[classification], initialized to smoothing for all values
		self.classProb = []
		for i in range(numClass):
			self.classProb.append(0)

		#for each element in the set get classification, then add +1 to each feature present (for binary)
		#additionally, increment frequency of classifications
		for i in range(len(sampleSet)):
			currClassification = sampleSet[i][0]
			self.classProb[currClassification] += 1
			for j in range(featureCount):
				self.featureProb[currClassification][j] += sampleSet[i][j+1]

		#normalize for self.featureProb
		for i in range(featureCount):
			total = 0
			for j in range(numClass):
				total += self.featureProb[i][j]
			for j in range(numClass):
				self.featureProb[i][j] /= total

		#normalize for self.classProb
		total = 0
		for i in range(numClass):
			total += self.classProb[i]
		for i in range(numClass):
			self.classProb[i] /= total

	def assess(test):
		if len(self.classProb) == 0:
			print("Classifier not initialized, please initialize first.\n")
		else:
			#preset the max value and the index of its classification to 0
			maxClass = 0
			maxVal = 0
			
			#initialize max value
			maxVal = math.log(self.classProb[0])
			for i in range(len(featureProb[0])):
				maxVal += math.log(self.featureProb[0][i])

			#iterate through possible classifications, selecting for max value
			for i in range(1, len(self.classProb)):
				temp = math.log(self.classProb[i])
				for j in range(len(featureProb[0])):
					maxVal += math.log(self.featureProb[i][j])
				if temp > maxVal:
					maxClass = i
					maxVal = temp
		return maxVal




