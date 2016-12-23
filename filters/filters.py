#!/usr/bin/python


import numpy as np

from util import normalize_input
from util import dot3
from util import inverse

RUN_VALIDATION = True
SAVE_DATA = True


class filter_error(Exception):
    def __str__(self):
        return 'something went wrong'



class process(object):
    __slots__ = ('dim', 'predict', 'noise', 'diff')

    def __init__(self, dim, predict, noise, *args):
        self.dim, self.noise = dim, noise
        self.predict = lambda x, dt : normalize_input(predict(x, dt))

        if args:
            self.diff = args[0]
        else:
            self.diff = None

    def has_diff(self):
        return self.diff is not None


class measure(object):
    __slots__ = ('dimx', 'dimz', 'measure', 'noise', 'diff')

    def __init__(self, dimx, dimz, measure, noise, *args):
        self.dimx, self.dimz, self.noise = dimx, dimz, noise
        self.measure = lambda x, dt : normalize_input(measure(x, dt))

        if args:
            self.diff = args[0]
        else:
            self.diff = None

    def has_diff(self):
        return self.diff is not None



class EKF(object):
    __slots__ = (
        '_process', '_measurement',
        'x', 'cov', '_time',
        '_I',
        '_y', '_S', '_K')

    def __init__(self, proc, meas):
        self._process, self._measurement = proc, meas

        if RUN_VALIDATION:
            self.__validate()

            if not self._process.has_diff():
                raise filter_error()

            if not self._measurement.has_diff():
                raise filter_error()
        self._time = None
        self._I = np.identity(self._process.dim, float)

    def __validate(self):
        if self._process.dim != self._measurement.dimx:
            raise filter_error()

    def set_initial(self, t, x, *args):
        self._time = t
        self.x = normalize_input(x)
        if args:
            self.cov = args[0]
        else:
            self.cov = np.identity(self._process.dim, float)

    def get_prediction(self, t):
        proc = self._process
        F, dF, Q = proc.predict, proc.diff, proc.noise

        dt = t - self._time
        x0 = np.reshape(self.x, (self._process.dim,))
        #print 'start from time', x0

        x1 = F(x0, dt)

        dF = dF(x0, dt)
        cov1 = dot3(dF, self.cov, dF.T) + Q(x0, dt)

        #print 'return value', x1
        return x1, cov1

    def update(self, t, data):
        meas = self._measurement
        H, dH, R = meas.measure, meas.diff, meas.noise

        #print '========================================================'
        #print 'at time', self._time
        #print 'current state', self.x

        self.x, self.cov = self.get_prediction(t)
        dt = t - self._time

        #print 'after prediction', self.x

        z = normalize_input(data)
        x = np.reshape(self.x, (self._process.dim,))
        dH = dH(x, dt)

        # Innovation vector
        y = z - H(x, dt)
        # Innovation covariance
        S = dot3(dH, self.cov, dH.T) + R(x, dt)
        # Near-optimal Kalman gain
        K = dot3(self.cov, dH.T, inverse(S))

        self.x += K.dot(y)

        self.cov = (self._I - K.dot(dH)).dot(self.cov)
        self._time = t

        if SAVE_DATA:
            self._y = y
            self._S = S
            self._K = K

        #print 'self.x', self.x
        return self.x, self.cov



class UKF(object):
    pass
