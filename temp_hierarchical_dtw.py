import numpy as np

THRESHOLD = 2
WEIGHT_FOR_NEXT_COL = 0.2

def fillWeightMatrix(shape, column):
	weightMatrix = np.zeros(shape)
	# TODO: add comment
	high = (1-WEIGHT_FOR_NEXT_COL)/(shape[1]-1)
	# The weight for everything in a column should be the same
	for col in range(shape[1]):
		for row in range(shape[0]):
			if col == column - 1:
				weightMatrix[row][col] = WEIGHT_FOR_NEXT_COL
			else:
				weightMatrix[row][col] = high

	return weightMatrix


def hierarchicalDTW_B(cost):
	height = cost.shape[0]
	width = cost.shape[1]

	value = np.empty((), dtype=object)
	value[()] = (0, 0)
	hierarchicalBacktrace = np.full((height, width), value, dtype=object)

	hierarchicalCost = np.zeros((height, width))

	for row in range(0, height):
		for col in range(0,width):
			
			# initialize bottom two rows as infinity
			if row == 0 or row == 1:
				hierarchicalCost[row][col] = float("inf")
				hierarchicalBacktrace[row][col] = (-1, -1)
			elif row == 1:
				# fill in the first column with cost from cost matrix, since it could map to first midi section
				if col == 0:
					hierarchicalCost[row][col] = cost[row][col][0]
					hierarchicalBacktrace[row][col] = (-1, -1)
				else:
					hierarchicalCost[row][col] = float("inf")
					hierarchicalBacktrace[row][col] = (-1, -1)
			
			else:
				startHeight = int(cost[row][col][1])

				# only look in matrix of "allowable" values, so at times before start time of current subseq
				subHierarchicalCost = np.copy(hierarchicalCost[max(0,startHeight-THRESHOLD):startHeight][0:])

				# fill in weight matrix, with size same as sub cost matrix
				weightMatrix = fillWeightMatrix(subHierarchicalCost.shape, col)

				# multiply W by cost of the current cell
				weightedCurrentCost = np.multiply(weightMatrix, cost[row][col][0])

				# cost of previous cells + W * cost of current cell, to calculate all possible costs
				fullCost = subHierarchicalCost + weightedCurrentCost


				# find the minimum cost in the matrix, and the location of this cost
				# TODO: what happens if multiple things have the same cost?
				minCost = float("inf")
				backtrace = (-1, -1)

				if fullCost != None and fullCost.size != 0:
					minCost = np.min(fullCost)
					backtrace = np.unravel_index(fullCost.argmin(), fullCost.shape)

				# need to adjust backtrace, since this is the index in only the subcost matrix
				backtrace = (backtrace[0] + startHeight - THRESHOLD, backtrace[1])

				# for the first strip, we want to also consider the cost of starting at this strip
				if col == 0 and startHeight < THRESHOLD:
					if cost[row][col][0] < minCost:
						minCost = cost[row][col][0]
						backtrace = (-1, -1)

				# update cost and backtrace matrices
				hierarchicalCost[row][col] = minCost
				hierarchicalBacktrace[row][col] = backtrace

	return hierarchicalCost, hierarchicalBacktrace

def hierarchicalDTW_B_sad(endCostCosts, endCostStartLocations):
	height = endCostCosts.shape[0]
	width = endCostCosts.shape[1]

	value = np.empty((), dtype=object)
	value[()] = (0, 0)
	hierarchicalBacktrace = np.full((height, width), value, dtype=object)

	hierarchicalCost = np.zeros((height, width))

	print(height)
	print(width)

	for row in range(0, height):
		print(row)
		for col in range(0,width):
			
			# initialize bottom two rows as infinity
			if row == 0 or row == 1:
				hierarchicalCost[row][col] = float("inf")
				hierarchicalBacktrace[row][col] = (-1, -1)
			elif row == 1:
				# fill in the first column with cost from cost matrix, since it could map to first midi section
				if col == 0:
					hierarchicalCost[row][col] = endCostCosts[row][col]
					hierarchicalBacktrace[row][col] = (-1, -1)
				else:
					hierarchicalCost[row][col] = float("inf")
					hierarchicalBacktrace[row][col] = (-1, -1)
			
			else:
				startHeight = int(endCostStartLocations[row][col])

				# only look in matrix of "allowable" values, so at times before start time of current subseq
				subHierarchicalCost = np.copy(hierarchicalCost[max(0,startHeight-THRESHOLD):startHeight][0:])

				# fill in weight matrix, with size same as sub cost matrix
				weightMatrix = fillWeightMatrix(subHierarchicalCost.shape, col)

				# multiply W by cost of the current cell
				weightedCurrentCost = np.multiply(weightMatrix, endCostCosts[row][col])

				# cost of previous cells + W * cost of current cell, to calculate all possible costs
				fullCost = subHierarchicalCost + weightedCurrentCost

				# find the minimum cost in the matric, and the location of this cost
				# TODO: what happens if multiple things have the same cost?
				minCost = np.min(fullCost)
				backtrace = np.unravel_index(fullCost.argmin(), fullCost.shape)

				# need to adjust backtrace, since this is the index in only the subcost matrix
				backtrace = (backtrace[0] + startHeight - THRESHOLD, backtrace[1])

				# for the first strip, we want to also consider the cost of starting at this strip
				if col == 0 and startHeight < THRESHOLD:
					if endCostCosts[row][col] < minCost:
						minCost = endCostCosts[row][col]
						backtrace = (-1, -1)

				# update cost and backtrace matrices
				hierarchicalCost[row][col] = minCost
				hierarchicalBacktrace[row][col] = backtrace

	return hierarchicalCost, hierarchicalBacktrace


# make test input
# cost = np.array([[(float("inf"),0),(float("inf"),0),(float("inf"),0)],
# 				[(1,0),(5,0),(5,0)],
# 				[(5,1),(5,1),(5,1)],
# 				[(5,2),(1,2),(5,2)],
# 				[(5,2),(5,2),(1,2)],
# 				[(5,4),(5,4),(1,4)]])


# hierarchicalCost, hierarchicalBacktrace = hierarchicalDTW_B(cost)


endCostCosts = np.loadtxt(open("endCostCosts.csv"), delimiter=",")
endCostStartLocations = np.loadtxt(open("endCostStartLocations.csv"), delimiter=",")

hierarchicalDTW_B_sad(endCostCosts, endCostStartLocations)

print(hierarchicalCost)
print(hierarchicalBacktrace)


