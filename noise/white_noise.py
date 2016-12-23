#!/usr/bin/python


import numpy.random as rnd


class noise_error(Exception):
    pass


class white_noise(object):

    __slots__ = ('_mean', '_sigma')

    def __init__(self, *args, **kwargs):
        self._mean, self._sigma = 0, None

        if kwargs:
            self._mean = kwargs.get('mean', None)
            self._sigma = kwargs.get('sigma', 0)

        elif args:
            self._sigma = args[0]
            if len(args) > 1:
                self._mean = args[1]

        if self._sigma is None:
            raise noise_error()


    def __call__(self):
        return rnd.normal(self._mean, self._sigma)
