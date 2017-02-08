#!/usr/bin/env/python
"""
Contains functions for finding the indices of
relative maxima and minima of an array
"""

def maxima(x, withends=False):
	"""
	Identify the indices of the maxima of an array.
	If the option withends=True is chosen, include endpoints.
	"""

	import numpy as np
	
	#Calculate sign of slope between every two points
	dx = np.zeros(len(x))
	dx[1:] = np.diff(x)
	dx[0] = dx[1]
	dx = np.sign(dx)

	#Calculate the concavity; places where the sign
	#of the derivative changes will show up as nonzero entries
	d2x = np.zeros(len(x))
	d2x[:-1] = np.diff(dx)

	#Take care of endpoints
	if withends:
		if x[0] > x[1]:
			d2x[0] = -2
		if x[-1] > x[-2]:
			d2x[-1] = -2

	#Identify nonzero entries of d2x that correspond to maxima
	indices = np.nonzero(d2x == -2)[0]

	return indices


def minima(x, withends=False):
	"""
	Identify the indices of the minima of an array.
	If the option withends=True is chosen, include endpoints.
	"""

	import numpy as np
	
	#Calculate sign of slope between every two points
	dx = np.zeros(len(x))
	dx[1:] = np.diff(x)
	dx[0] = dx[1]
	dx = np.sign(dx)

	#Calculate the concavity; places where the sign
	#of the derivative changes will show up as nonzero entries
	d2x = np.zeros(len(x))
	d2x[:-1] = np.diff(dx)

	#Take care of endpoints
	if withends:
		if x[0] < x[1]:
			d2x[0] = 2
		if x[-1] < x[-2]:
			d2x[-1] = 2

	#Identify nonzero entries of d2x that correspond to minima
	indices = np.nonzero(d2x == 2)[0]

	return indices

