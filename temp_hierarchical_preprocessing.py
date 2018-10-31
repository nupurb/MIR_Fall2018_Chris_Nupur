import numpy as np

np.set_printoptions(threshold=np.nan)

def preprocessHierarchicalDTW(hierarchicalDTWStartLocations):
	height = hierarchicalDTWStartLocations.shape[0]
	width = hierarchicalDTWStartLocations.shape[1]

	endOfNextPath = np.matrix(np.ones((height, width)) * np.inf)

	# Fill in the row index of the end of the first bridge we see
	# where the first bridge we see has a start height greater than the 
	# row we're at
	for col in range(width):
		savedEnd = -1
		savedStart = -1

		for row in range(height):
			# if we have a saved end value, fill that in
			if row < savedStart:
				endOfNextPath[row, col] = savedEnd
			else:
				# look through the hierarchical dtw start locations until we find the first branch
				done = False
				startLocIndex = row+1

				while not done and startLocIndex < height:
					if hierarchicalDTWStartLocations[startLocIndex][col] > row:

						endOfNextPath[row, col] = startLocIndex

						savedStart = hierarchicalDTWStartLocations[startLocIndex][col]
						savedEnd = startLocIndex

						done = True
					startLocIndex += 1

	return endOfNextPath

hierarchicalDTWStartLocations = np.loadtxt(open("endCostStartLocations.csv"), delimiter=",")

endOfNextPath = preprocessHierarchicalDTW(hierarchicalDTWStartLocations)

print(endOfNextPath)


