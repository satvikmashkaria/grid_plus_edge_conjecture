def dist(x, y, a, b):
	return abs(x-a)+abs(y-b)

def distf(x, y, a, b, x1, y1, x2, y2):
	"""
	distance between (x, y) and (a, b) with extra edge between (x1, y1) and (x2, y2).
	"""
	return min(dist(a, b, x, y), dist(x, y, x1, y1)+dist(x2, y2, a, b)+1, dist(x, y, x2, y2)+dist(x1, y1, a, b)+1)

def four_checker(x1, y1, x2, y2, n, m):
	"""
	Condition for which MD will be 4 according to the conjecture.
	"""
	cornerset = [(1,1), (n,m), (1,m), (n,1)]
	Gain1 = abs(abs(y2 - y1) - abs(x2 - x1)) - 1

	first = ((x1, y1) not in cornerset) and ((x2, y2) not in cornerset)
	second = Gain1 % 2 == 0 and Gain1 > 0
	third = (min(abs(x2 - x1), abs(y2 - y1)) >= (Gain1/2) + 2)
	return  first and second and third

def two_checker(x1, y1, x2, y2, n, m):
	"""
	Condition for which MD will be 2 according to the conjecture.
	"""
	cornerset = [(1,1), (n,m), (1,m), (n,1)]
	Gain1 = abs(abs(y2 - y1) - abs(x2 - x1)) - 1
	Gain = abs(y2 - y1) + abs(x2 - x1) - 1

	first = Gain == 1
	second = (((x1,y1) in cornerset or (x2, y2) in cornerset) and (Gain1 <= 1) and (Gain%2 == 1))
	third = (((x1,y1) in cornerset or (x2, y2) in cornerset) and (Gain1 >= 3) and (Gain%2 == 1) and (Gain - Gain1 <= 2))
	fourth = (((x1,y1) in cornerset and (x2, y2) in cornerset)) and Gain%2 == 1
	return first or second or third or fourth