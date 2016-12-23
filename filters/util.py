#!/usr/bin/python


import numpy as np


def dot3(A, B, C):
	return A.dot(B.dot(C))


def normalize_input(data):
	if isinstance(data, np.ndarray):
		shape = data.shape
		if len(shape) == 1:
			return np.reshape(data.astype(float), (shape[0], 1))
		else:
			return data.astype(float)
	else:
		length = len(data)
		return np.reshape(np.array(data, float), (length, 1))


def inverse(A):
	return np.linalg.inv(A)


if __name__ == '__main__':
	def test_1():
		data = np.array((1, 2), float)
		out = normalize_input(data)
		assert np.array_equal(out, np.array(((1,), (2,)), float))

	def test_2():
		data = (1, 2)
		out = normalize_input(data)
		assert np.array_equal(out, np.array(((1,), (2,)), float))

	def test_3():
		data = np.array(((1), (2)))
		out = normalize_input(data)
		assert np.array_equal(out, np.array(((1,), (2,)), float))



	test_1()
	test_2()
	test_3()
