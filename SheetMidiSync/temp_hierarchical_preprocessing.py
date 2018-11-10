import numpy as np
import matplotlib.pyplot as plt


np.set_printoptions(threshold=np.nan)

def hierarchicalDTW(hierarchicalCost, hierarchicalDTWStartLocations, endOfNextPath):
	'''
	We assume we have N strips, and a bootleg MIDI M pixels long

	hierarchicalCost: MxN np array. The last costs in each subsequence DTW accumulated cost matrix concatenated together
	hierarchicalDTWStartLocations: MxN np array. For location m, n it contains the start row of the path that ends at location m, n.
	endOfNextPath: MxN np array. For location m, n it contains the end row of the next path in strip n that starts strictly above location m (as if your previous path ended at m). 

	Returns:
	hierarchicalAccumulatedValue: our accumulated value matrix
	backtrace: an MxN np array of (int, int) tuples. 
	''' 

	INT_INF = np.array([np.inf]).astype(int)[0] # get the int that np.inf is automatically cast to

	hierarchicalValue = -hierarchicalCost

	height = hierarchicalCost.shape[0]
	width = hierarchicalCost.shape[1]

	WEIGHT_FOR_NEXT_COL = 1.0
	# TEMPORARY make it impossible to jump to other cols
	#WEIGHT_FOR_OTHER_COLS = (1.0 - WEIGHT_FOR_NEXT_COL) / (width - 1)
	WEIGHT_FOR_OTHER_COLS = -np.inf

	# initialize an empty backtrace matrix
	value = np.empty((), dtype=object)
	value[()] = (-1, -1)
	hierarchicalBacktrace = np.full((height, width), value, dtype=object)

	# Initialize all value to be -inf
	accumHierarchicalValue = np.full((height, width), -np.inf)

	# initialize the first column
	# TEMPORARY: only allow us to start at (0,0)
	# accumHierarchicalValue[:, 0] = hierarchicalValue[:, 0]
	# so fill in only the rows which have a 0 start location
	for row in range(height):
		if hierarchicalDTWStartLocations[row, 0] == 0:
			print("Filling row ", row)
			accumHierarchicalValue[row, 0] = hierarchicalValue[row, 0]

	print("after first col: ", accumHierarchicalValue[:,0])

	# Consider jumps from every location (except in the last row)
	for row in range(height - 1):
		print("Percent Done: ", round(100 * (float(row) / height), 1))
		for col in range(width):
			# update all locations that a path ending at this location can jump to.
			# for now we can't jump to our own column, so we remove it
			possibleCols = list(range(width))
			possibleCols.remove(col)
			for jumpCol in possibleCols:
				jumpRow = endOfNextPath[row, jumpCol]
				if (not jumpRow == INT_INF):
					# weight transitioning to the next strip higher than jumping
					curWeight = WEIGHT_FOR_NEXT_COL if (jumpCol == col + 1) else WEIGHT_FOR_OTHER_COLS
					newValue = accumHierarchicalValue[row, col] + curWeight * hierarchicalValue[jumpRow, jumpCol]
					# if we have found a higher value path to jumpRow, jumpCol then update our accumValue and backtrace
					if (newValue > accumHierarchicalValue[jumpRow, jumpCol]):
						# store the new value and the location you jumped from
						accumHierarchicalValue[jumpRow, jumpCol] = newValue
						hierarchicalBacktrace[jumpRow, jumpCol] = (row, col)
			
			# always consider the option to head directly upwards with 0 value added
			# TEMPORARY: for now ignore the option right upwards

			#if (accumHierarchicalValue[row, col] > accumHierarchicalValue[row + 1, col]):
				#accumHierarchicalValue[row + 1, col] = accumHierarchicalValue[row, col]
				#hierarchicalBacktrace[row + 1, col] = (row, col)

	return accumHierarchicalValue, hierarchicalBacktrace


def hierarchicalBacktrace(accumHierarchicalValue, hierarchicalBacktrace):
	height = accumHierarchicalValue.shape[0]
	width = accumHierarchicalValue.shape[1]
	curBest = -np.inf
	curRow = height - 1
	while (curBest == -np.inf):
		print("cur row is ", curRow)
		curBest = accumHierarchicalValue[curRow, -1]
		curRow = curRow - 1
	curRow = curRow + 1
	curCol = width - 1
	#curRow = height - 1
	print(curCol)
	print(curRow)
	print("Cur Row: ", curRow, " Cur Col: ", curCol)
	while ((curRow, curCol) != (0, 0) and hierarchicalBacktrace[curRow, curCol] != (-1, -1)):
		print("Cur Strip: ", curCol, " Cur Row: ", curRow, " Cur Value: ", accumHierarchicalValue[curRow, curCol], " Accum Value: ", accumHierarchicalValue[curRow, :])
		curRow, curCol = hierarchicalBacktrace[curRow, curCol]
	print("Cur Strip: ", curCol, " Cur Row: ", curRow, " Cur Value: ", accumHierarchicalValue[curRow, curCol], " Accum Value: ", accumHierarchicalValue[curRow, :])

def hierarchicalBacktraceNotEnd(accumHierarchicalValue, hierarchicalBacktrace):
	height = accumHierarchicalValue.shape[0]
	width = accumHierarchicalValue.shape[1]
	curBest = -np.inf
	curRow = height - 1
	curCol = width - 1
	while (curBest == -np.inf):
		curBest = np.amax(accumHierarchicalValue[curRow, :])
		curCol = np.argmax(accumHierarchicalValue[curRow, :])
		curRow = curRow - 1
	curRow = curRow + 1
	#curRow = height - 1
	print("Cur Row: ", curRow, " Cur Col: ", curCol)
	while ((curRow, curCol) != (0, 0) and hierarchicalBacktrace[curRow, curCol] != (-1, -1)):
		print("Cur Strip: ", curCol, " Cur Row: ", curRow, " Cur Value: ", accumHierarchicalValue[curRow, curCol], " Accum Value: ", accumHierarchicalValue[curRow, :])
		curRow, curCol = hierarchicalBacktrace[curRow, curCol]
	print("Cur Strip: ", curCol, " Cur Row: ", curRow, " Cur Value: ", accumHierarchicalValue[curRow, curCol], " Accum Value: ", accumHierarchicalValue[curRow, :])


def oldHierarchicalBacktrace(accumHierarchicalValue, hierarchicalBacktrace):
	height = accumHierarchicalValue.shape[0]
	width = accumHierarchicalValue.shape[1]

	curCol = np.argmax(accumHierarchicalValue[-1, :])
	curRow = height - 1
	print(curCol)
	print(curRow)

	while ((curRow, curCol) != (0, 0) and hierarchicalBacktrace[curRow, curCol] != (-1, -1)):
		print("Cur Strip: ", curCol, " Cur Row: ", curRow, " Cur Value: ", accumHierarchicalValue[curRow, curCol], " Accum Value: ", accumHierarchicalValue[curRow, :])
		curRow, curCol = hierarchicalBacktrace[curRow, curCol]
	print("Cur Strip: ", curCol, " Cur Row: ", curRow, " Cur Value: ", accumHierarchicalValue[curRow, curCol], " Accum Value: ", accumHierarchicalValue[curRow, :])


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

	return endOfNextPath.astype(int)

hierarchicalCost = np.loadtxt(open("mazurkaEndCostCosts.csv"), delimiter=",")
hierarchicalDTWStartLocations = np.loadtxt(open("mazurkaEndCostStartLocations.csv"), delimiter=",")

'''
# test case
hierarchicalCost = np.full((14, 4), np.inf)
hierarchicalCost[3, 0] = -1000
hierarchicalCost[5, 1] = -1000
hierarchicalCost[9, 1] = -1000
hierarchicalCost[7, 2] = -1000
hierarchicalCost[11, 2] = -1000
hierarchicalCost[13, 3] = - 1000
print(hierarchicalCost)

endOfNextPath = np.full((14, 4), np.inf)
endOfNextPath[0:2,0]=3
endOfNextPath[0:4,1]=5
endOfNextPath[4:8,1]=9
endOfNextPath[0:6,2]=7
endOfNextPath[6:10,2]=11
endOfNextPath[0:,3]=13
print(endOfNextPath)
'''

endOfNextPath = preprocessHierarchicalDTW(hierarchicalDTWStartLocations)

accumCost, backtrace = hierarchicalDTW(hierarchicalCost, hierarchicalDTWStartLocations, endOfNextPath.astype(int))
hierarchicalBacktrace(accumCost, backtrace)
