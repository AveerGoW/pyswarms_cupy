# -*- coding: utf-8 -*-

"""single_obj.py: collection of single-objective functions

All objective functions :code:`obj_func()` must accept a
:code:`(numpy.ndarray)` with shape :code:`(n_particles, dimensions)`.
Thus, each row represents a  particle, and each column represents its
position on a specific dimension of the search-space.

In this context, :code:`obj_func()` must return an array :code:`j`
of size :code:`(n_particles, )` that contains all the computed fitness
for each particle.

Whenever you make changes to this file via an implementation
of a new objective function, be sure to perform unittesting
in order to check if all functions implemented adheres to
the design pattern stated above.

Function list:
- Ackley's, ackley
- Beale, beale
- Booth, booth
- Bukin's No 6, bukin6
- Cross-in-Tray, crossintray
- Easom, easom
- Eggholder, eggholder
- Goldstein, goldstein
- Himmelblau's, himmelblau
- Holder Table, holdertable
- Levi, levi
- Matyas, matyas
- Rastrigin, rastrigin
- Rosenbrock, rosenbrock
- Schaffer No 2, schaffer2
- Sphere, sphere
- Three Hump Camel, threehump
"""

# Import modules
import numpy as np
import cupy as cp


def ackley(x):
    """Ackley's objective function.

    Has a global minimum of `0` at :code:`f(0,0,...,0)` with a search
    domain of [-32, 32]

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`


    ------
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not cp.logical_and(x >= -32, x <= 32).all():                            #changed np to cp
        raise ValueError("Input for Ackley function must be within [-32, 32].")

    d = x.shape[1]
    j = (
        -20.0 * cp.exp(-0.2 * cp.sqrt((1 / d) * (x ** 2).sum(axis=1)))         #changed np to cp
        - cp.exp((1 / float(d)) * cp.cos(2 * cp.pi * x).sum(axis=1))           #changed np to cp
        + 20.0
        + cp.exp(1)                                                            #changed np to cp
    )

    return j


def beale(x):
    """Beale objective function.

    Only takes two dimensions and has a global minimum of `0` at
    :code:`f([3,0.5])` Its domain is bounded between :code:`[-4.5, 4.5]`

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    Raises
    ------
    IndexError
        When the input dimensions is greater than what the function
        allows
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not x.shape[1] == 2:
        raise IndexError("Beale function only takes two-dimensional input.")
    if not cp.logical_and(x >= -4.5, x <= 4.5).all():                        #changed np to cp
        raise ValueError(
            "Input for Beale function must be within " "[-4.5, 4.5]."
        )

    x_ = x[:, 0]
    y_ = x[:, 1]
    j = (
        (1.5 - x_ + x_ * y_) ** 2.0
        + (2.25 - x_ + x_ * y_ ** 2.0) ** 2.0
        + (2.625 - x_ + x_ * y_ ** 3.0) ** 2.0
    )

    return j


def booth(x):
    """Booth's objective function.

    Only takes two dimensions and has a global minimum of `0` at
    :code:`f([1,3])`. Its domain is bounded between :code:`[-10, 10]`

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    Raises
    ------
    IndexError
        When the input dimensions is greater than what the function
        allows
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not x.shape[1] == 2:
        raise IndexError("Booth function only takes two-dimensional input.")
    if not cp.logical_and(x >= -10, x <= 10).all():                            #changed np to cp
        raise ValueError("Input for Booth function must be within [-10, 10].")

    x_ = x[:, 0]
    y_ = x[:, 1]
    j = (x_ + 2 * y_ - 7) ** 2.0 + (2 * x_ + y_ - 5) ** 2.0

    return j


def bukin6(x):
    """Bukin N. 6 Objective Function

    Only takes two dimensions and has a global minimum  of `0` at
    :code:`f([-10,1])`. Its coordinates are bounded by:
        * x[:,0] must be within [-15, -5]
        * x[:,1] must be within [-3, 3]

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    Raises
    ------
    IndexError
        When the input dimensions is greater than what the function
        allows
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not x.shape[1] == 2:
        raise IndexError(
            "Bukin N. 6 function only takes two-dimensional " "input."
        )
    if not cp.logical_and(x[:, 0] >= -15, x[:, 0] <= -5).all():             #changed np to cp
        raise ValueError(
            "x-coord for Bukin N. 6 function must be within " "[-15, -5]."
        )
    if not cp.logical_and(x[:, 1] >= -3, x[:, 1] <= 3).all():               #changed np to cp
        raise ValueError(
            "y-coord for Bukin N. 6 function must be within " "[-3, 3]."
        )

    x_ = x[:, 0]
    y_ = x[:, 1]
    j = 100 * cp.sqrt(cp.absolute(y_ - 0.01 * x_ ** 2.0)) + 0.01 * cp.absolute(    #changed np to cp
        x_ + 10
    )

    return j


def crossintray(x):
    """Cross-in-tray objective function.

    Only takes two dimensions and has a four equal global minimums
     of `-2.06261` at :code:`f([1.34941, -1.34941])`, :code:`f([1.34941, 1.34941])`,
     :code:`f([-1.34941, 1.34941])`, and :code:`f([-1.34941, -1.34941])`.
    Its coordinates are bounded within :code:`[-10,10]`.

    Best visualized in the full domain and a range of :code:`[-2.0, -0.5]`.

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    Raises
    ------
    IndexError
        When the input dimensions is greater than what the function
        allows
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not x.shape[1] == 2:
        raise IndexError(
            "Cross-in-tray function only takes two-dimensional input."
        )
    if not cp.logical_and(x >= -10, x <= 10).all():                        #changed np to cp
        raise ValueError(
            "Input for cross-in-tray function must be within [-10, 10]."
        )

    x_ = x[:, 0]
    y_ = x[:, 1]

    j = -0.0001 * cp.power(                                            #changed np to cp
        cp.abs(                                                        #changed np to cp
            cp.sin(x_)                                                 #changed np to cp
            * cp.sin(y_)                                               #changed np to cp
            * cp.exp(cp.abs(100 - (cp.sqrt(x_ ** 2 + y_ ** 2) / cp.pi)))    #changed np to cp
        )
        + 1,
        0.1,
    )

    return j


def easom(x):
    """Easom objective function.

    Only takes two dimensions and has a global minimum of
    `-1` at :code:`f([pi, pi])`.
    Its coordinates are bounded within :code:`[-100,100]`.

    Best visualized in the domain of :code:`[-5, 5]` and a range of :code:`[-1, 0.2]`.

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    Raises
    ------
    IndexError
        When the input dimensions is greater than what the function
        allows
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not x.shape[1] == 2:
        raise IndexError("Easom function only takes two-dimensional input.")
    if not cp.logical_and(x >= -100, x <= 100).all():                        #changed np to cp
        raise ValueError(
            "Input for Easom function must be within [-100, 100]."
        )

    x_ = x[:, 0]
    y_ = x[:, 1]

    j = (
        -1
        * cp.cos(x_)                                            #changed np to cp
        * cp.cos(y_)                                            #changed np to cp
        * cp.exp(-1 * ((x_ - cp.pi) ** 2 + (y_ - cp.pi) ** 2))  #changed np to cp
    )

    return j


def eggholder(x):
    """Eggholder objective function.

    Only takes two dimensions and has a global minimum of
    `-959.6407` at :code:`f([512, 404.3219])`.
    Its coordinates are bounded within :code:`[-512, 512]`.

    Best visualized in the full domain and a range of :code:`[-1000, 1000]`.

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    Raises
    ------
    IndexError
        When the input dimensions is greater than what the function
        allows
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not x.shape[1] == 2:
        raise IndexError(
            "Eggholder function only takes two-dimensional input."
        )
    if not cp.logical_and(x >= -512, x <= 512).all():                #changed np to cp
        raise ValueError(
            "Input for Eggholder function must be within [-512, 512]."
        )

    x_ = x[:, 0]
    y_ = x[:, 1]

    j = -(y_ + 47) * cp.sin(cp.sqrt(cp.abs((x_ / 2) + y_ + 47))) - x_ * cp.sin(    #changed np to cp
        cp.sqrt(cp.abs(x_ - (y_ + 47)))                                            #changed np to cp
    )

    return j


def goldstein(x):
    """Goldstein-Price's objective function.

    Only takes two dimensions and has a global minimum at
    :code:`f([0,-1])`. Its domain is bounded between :code:`[-2, 2]`

    Best visualized in the domain of :code:`[-1.3,1.3]` and range :code:`[-1,8000]`

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    Raises
    ------
    IndexError
        When the input dimensions is greater than what the function
        allows
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not x.shape[1] == 2:
        raise IndexError(
            "Goldstein function only takes two-dimensional " "input."
        )
    if not cp.logical_and(x >= -2, x <= 2).all():                        #changed np to cp
        raise ValueError(
            "Input for Goldstein-Price function must be within " "[-2, 2]."
        )

    x_ = x[:, 0]
    y_ = x[:, 1]
    j = (
        1
        + (x_ + y_ + 1) ** 2.0
        * (
            19
            - 14 * x_
            + 3 * x_ ** 2.0
            - 14 * y_
            + 6 * x_ * y_
            + 3 * y_ ** 2.0
        )
    ) * (
        30
        + (2 * x_ - 3 * y_) ** 2.0
        * (
            18
            - 32 * x_
            + 12 * x_ ** 2.0
            + 48 * y_
            - 36 * x_ * y_
            + 27 * y_ ** 2.0
        )
    )

    return j


def himmelblau(x):
    """Himmelblau's  objective function

    Only takes two dimensions and has a four equal global minimums
     of zero at :code:`f([3.0,2.0])`, :code:`f([-2.805118,3.131312])`,
     :code:`f([-3.779310,-3.283186])`, and :code:`f([3.584428,-1.848126])`.
    Its coordinates are bounded within :code:`[-5,5]`.

    Best visualized with the full domain and a range of :code:`[0,1000]`

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    Raises
    ------
    IndexError
        When the input dimensions is greater than what the function
        allows
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not x.shape[1] == 2:
        raise IndexError(
            "Himmelblau function only takes two-dimensional input."
        )
    if not cp.logical_and(x >= -5, x <= 5).all():                    #changed np to cp
        raise ValueError(
            "Input for Himmelblau function must be within [-5,5]."
        )

    x_ = x[:, 0]
    y_ = x[:, 1]

    j = (x_ ** 2 + y_ - 11) ** 2 + (x_ + y_ ** 2 - 7) ** 2

    return j


def holdertable(x):
    """Holder Table objective function

    Only takes two dimensions and has a four equal global minimums
     of `-19.2085` at :code:`f([8.05502, 9.66459])`, :code:`f([-8.05502, 9.66459])`,
     :code:`f([8.05502, -9.66459])`, and :code:`f([-8.05502, -9.66459])`.
    Its coordinates are bounded within :code:`[-10, 10]`.

    Best visualized with the full domain and a range of :code:`[-20, 0]`

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    Raises
    ------
    IndexError
        When the input dimensions is greater than what the function
        allows
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not x.shape[1] == 2:
        raise IndexError(
            "Holder Table function only takes two-dimensional input."
        )
    if not cp.logical_and(x >= -10, x <= 10).all():                    #changed np to cp
        raise ValueError(
            "Input for Holder Table function must be within [-10,10]."
        )

    x_ = x[:, 0]
    y_ = x[:, 1]

    j = -cp.abs(                                                    #changed np to cp
        cp.sin(x_)                                                  #changed np to cp
        * cp.cos(y_)                                                #changed np to cp
        * cp.exp(cp.abs(1 - cp.sqrt(x_ ** 2 + y_ ** 2) / cp.pi))    #changed np to cp
    )

    return j


def levi(x):
    """Levi objective function

    Only takes two dimensions and has a global minimum at
    :code:`f([1,1])`. Its coordinates are bounded within
    :code:`[-10,10]`.

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    Raises
    ------
    IndexError
        When the input dimensions is greater than what the function
        allows
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not x.shape[1] == 2:
        raise IndexError("Levi function only takes two-dimensional input.")
    if not cp.logical_and(x >= -10, x <= 10).all():                            #changed np to cp
        raise ValueError("Input for Levi function must be within [-10, 10].")

    mask = cp.full(x.shape, False)            #changed np to cp
    mask[:, -1] = True
    masked_x = cp.ma.array(x, mask=mask)      #changed np to cp

    w_ = 1 + (x - 1) / 4
    masked_w_ = cp.ma.array(w_, mask=mask)    #changed np to cp
    d_ = x.shape[1] - 1

    j = (
        cp.sin(np.pi * w_[:, 0]) ** 2.0                                        #changed np to cp
        + ((masked_x - 1) ** 2.0).sum(axis=1)
        * (1 + 10 * cp.sin(cp.pi * (masked_w_).sum(axis=1) + 1) ** 2.0)        #changed np to cp
        + (w_[:, d_] - 1) ** 2.0 * (1 + cp.sin(2 * cp.pi * w_[:, d_]) ** 2.0)  #changed np to cp
    )

    return j


def matyas(x):
    """Matyas objective function

    Only takes two dimensions and has a global minimum at
    :code:`f([0,0])`. Its coordinates are bounded within
    :code:`[-10,10]`.

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
    """
    if not x.shape[1] == 2:
        raise IndexError("Matyas function only takes two-dimensional input.")
    if not cp.logical_and(x >= -10, x <= 10).all():                            #changed np to cp
        raise ValueError(
            "Input for Matyas function must be within " "[-10, 10]."
        )

    x_ = x[:, 0]
    y_ = x[:, 1]
    j = 0.26 * (x_ ** 2.0 + y_ ** 2.0) - 0.48 * x_ * y_

    return j


def rastrigin(x):
    """Rastrigin objective function.

    Has a global minimum at :code:`f(0,0,...,0)` with a search
    domain of :code:`[-5.12, 5.12]`

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    Raises
    ------
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not cp.logical_and(x >= -5.12, x <= 5.12).all():                    #changed np to cp
        raise ValueError(
            "Input for Rastrigin function must be within " "[-5.12, 5.12]."
        )

    d = x.shape[1]
    j = 10.0 * d + (x ** 2.0 - 10.0 * cp.cos(2.0 * cp.pi * x)).sum(axis=1)    #changed np to cp

    return j


def rosenbrock(x):
    """Rosenbrock objective function.

    Also known as the Rosenbrock's valley or Rosenbrock's banana
    function. Has a global minimum of :code:`np.ones(dimensions)` where
    :code:`dimensions` is :code:`x.shape[1]`. The search domain is
    :code:`[-inf, inf]`.

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`
    """

    r = cp.sum(                #changed np to cp
        100 * (x.T[1:] - x.T[:-1] ** 2.0) ** 2 + (1 - x.T[:-1]) ** 2.0, axis=0
    )

    return r


def schaffer2(x):
    """Schaffer N.2 objective function

    Only takes two dimensions and has a global minimum at
    :code:`f([0,0])`. Its coordinates are bounded within
    :code:`[-100,100]`.

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    Raises
    ------
    IndexError
        When the input dimensions is greater than what the function
        allows
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not x.shape[1] == 2:
        raise IndexError(
            "Schaffer N. 2 function only takes two-dimensional " "input."
        )
    if not cp.logical_and(x >= -100, x <= 100).all():                    #changed np to cp
        raise ValueError(
            "Input for Schaffer function must be within " "[-100, 100]."
        )

    x_ = x[:, 0]
    y_ = x[:, 1]
    j = 0.5 + (
        (cp.sin(x_ ** 2.0 - y_ ** 2.0) ** 2.0 - 0.5)        #changed np to cp
        / ((1 + 0.001 * (x_ ** 2.0 + y_ ** 2.0)) ** 2.0)
    )

    return j


def sphere(x):
    """Sphere objective function.

    Has a global minimum at :code:`0` and with a search domain of
        :code:`[-inf, inf]`

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`
    """
    j = (x ** 2.0).sum(axis=1)

    return j


def threehump(x):
    """Three-hump camel objective function

    Only takes two dimensions and has a global minimum of `0` at
    :code:`f([0, 0])`. Its coordinates are bounded within
    :code:`[-5, 5]`.

    Best visualized in the full domin and a range of :code:`[0, 2000]`.

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    Raises
    ------
    IndexError
        When the input dimensions is greater than what the function
        allows
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not x.shape[1] == 2:
        raise IndexError(
            "Three-hump camel function only takes two-dimensional input."
        )
    if not cp.logical_and(x >= -5, x <= 5).all():                        #changed np to cp
        raise ValueError(
            "Input for Three-hump camel function must be within [-5, 5]."
        )

    x_ = x[:, 0]
    y_ = x[:, 1]

    j = 2 * x_ ** 2 - 1.05 * (x_ ** 4) + (x_ ** 6) / 6 + x_ * y_ + y_ ** 2

    return j
