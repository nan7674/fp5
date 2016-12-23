#!/usr/bin/python


from numpy import array, sin, cos
from filters import process, measure, EKF


from filterpy.kalman import ExtendedKalmanFilter as rEKF

from noise import white_noise


S = 0.01

def model(state, dt):
    x, v = state
    return (x + v * dt, v)


def diff_model(state, dt):
    return array([
        [1., dt],
        [0., 1.]])

def model_noise(state, dt):
    t1 = dt
    t2 = t1 * t1
    t3 = t2 * t1

    return array([[S, 0], [0, S]])

    return S * array([
        [t3 / 3., t2 / 2.],
        [t2 / 2., t1     ]])


def model_measure(state, *args):
    x, _ = state
    return array([[x]], float)


def diff_model_measure(state, *args):
    return array([[1., 0.]])


def model_measure_noise(state, *args):
    return array([[0.1]])


def main():
    proc = process(
        2,
        model,
        model_noise,
        diff_model)

    meas = measure(
        2,
        1,
        model_measure,
        model_measure_noise,
        diff_model_measure)

    ekf = EKF(proc, meas)
    ekf.set_initial(0, (0., 1.))

    t, dt = 0, 0.1

    f = rEKF(dim_x=2, dim_z=1)
    f.P = array([[500., 0.], [0., 500.]])
    f.Q *= S
    f.x = array((0., 1.))
    f.F = array([[1., dt], [0., 1.]])
    f.H = array([[1., 0.]])
    f.R = 0.1

    F, dF = lambda t : sin(t), lambda x : cos(t)


    def M(x):
        return x[0]

    N = white_noise(0.3)

    while t < 10:

        f.predict(dt)
        XX, _ = ekf.get_prediction(t)

        V = F(t) + N()
        m = (V,)
        print t, F(t), V, XX[0][0], f.x[0]


        ekf.update(t, m)
        f.update(V, diff_model_measure, M)

        t += dt



if __name__ == '__main__':
    main()
