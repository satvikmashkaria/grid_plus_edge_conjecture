from utils import dist, distf, four_checker

import itertools
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

if len(sys.argv) < 3:
	print("Please provide dimensions of the grid.")
	exit(1)
n = int(sys.argv[1])
m = int(sys.argv[2])

point_list = itertools.product(range(1, n+1), range(1, m+1))
point_list = list(point_list)

def check_four(x1, y1, x2, y2):

	found = True

	for ((a1, b1), (a2, b2), (a3, b3)) in itertools.combinations(point_list, 3):
		
		distances = {} # Using dict for better search complexity
		flag = True

		for (i,j) in point_list:
			tup = (distf(i, j, a1, b1, x1, y1, x2, y2), distf(i, j, a2, b2, x1, y1, x2, y2), distf(i, j, a3, b3, x1, y1, x2, y2))
			if tup in distances:
				flag = False
				break
			else:
				distances[tup] = 1
		if flag == True:
			found = False
			break
		
	cond = four_checker(x1, y1, x2, y2, n, m)

	if found ^ cond == False:
	# Checking if XOR is false so no false positive or no false negatives.
		if found == True:
			print("MD is 4 when edge is between", str((x1, y1)), "and", str((x2, y2)))
	else:
		print("Mistake in ", x1, y1, x2, y2)
		exit()



t0 = time.time()
with ProcessPoolExecutor() as executor:
	for ((x1, y1), (x2, y2)) in itertools.combinations(point_list, 2):	
		if dist(x1, y1, x2, y2) > 1 and (x1 <= x2) and (y1 >= y2) and (y1 - y2) >= (x2 - x1):
			executor.submit(check_four, x1, y1, x2, y2)

print("Success. Conjecture for", n, "X", m, "grid is verified when MD is 4.")
t1 = time.time()		
print("Time taken: ", t1 - t0)