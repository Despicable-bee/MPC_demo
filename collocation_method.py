# Standard libs
from math import nan
from typing import List
from typing import Tuple
import sys
import math

# Third party libs
import jax.numpy as np
import jax
import numpy as onp
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)

import scipy.sparse.linalg as linalg

# Local libs

logger = logging.getLogger('MPC Collocation method JAX')

# * SYSTEM FUNCTIONS -----------------------------------------------------------

def invdx(x1: float, y1: float, x2: float, y2: float, stepSize: float):
    """ Inverse system model equation (x direction). 
    
    This function is used to determine what the controls would be from two sets
        of points and a step size.

    ARGS:
    - x1: x-coordinate of the first point
    - y1: y-coordinate of the first point
    - x2: x-coordinate of the second point
    - y2: y-coordinate of the second point

    RETURNS:
    - The control value.
    """

    r2 = x1**2 + y1**2
    xdot = ((y2 - y1) / (x2 - x1)) / stepSize

    return (xdot * (x2 + y2**2 / x2) + r2**2 + (y2*r2/x2))/r2

def convert_x_y_to_radius(x: float, y: float):
    return np.sqrt(x**2 + y**2)

def dx(x: float, y: float, uk: float):
    """ System model equation (x direction).

    This function defines part of the system of equations we want to control.
    
    ARGS:
    - x: The current position along the x-axis
    - y: The current position along the y-axis
    - uk: The current control value being applied (which is the value r)

    RETURNS:
    - The gradient of the system at point (x,y) in the x-direction.
    """
    r2 = x**2 + y**2
    xdot = (r2*uk - r2**2 - y*r2/x)/(x + y**2/x)
    return xdot

def dy(x: float, y: float, uk: float):
    """ System model equation (y direction).

    This function defines part of the system of equations we want to control.
    
    ARGS:
    - x: The current position along the x-axis
    - y: The current position along the y-axis
    - uk: The current control value being applied (which is the value r**2)

    RETURNS:
    - The gradient of the system at point (x,y) in the y direction.
    """
    r2 = x**2 + y**2
    xdot = dx(x,y,uk)
    ydot = (r2 + y*xdot)/x
    return ydot

# * LOSS FUNCTION --------------------------------------------------------------

@jax.jit
def loss_function(params: List[Tuple[float, float]], xr: float, ur: float,  
            stepSize: float,
            Qlist: List[float], Rlist: List[float]):
    """
    Computes the cost of the prediction.

    NOTE: No value for prediction horizon is given since the prediction horizon
        is implied by the step size + number of params.

    ARGS:
    - params: The control inputs to be optimised / evaluated (excludes inital
            control value).
    - xr: Target state value
    - ur: Resting control value
    - x0: Initial state values (the initial position (x,y))
    - u0: Initial control value
    - stepSize: The stepsize for the simulation
    - Qlist: weight values for the state change at each step
    - Rlist: weight values for the control change at each step

    RETURNS:
    - Scalar loss value
    """
    nanCorrector = 10e-7

    # Set initial parameters
    cost = 0

    # Generate the controls from the state points
    for k in range(1, len(params)):
        # Compute step
        xk = params[k-1]
        xkp1 = params[k]

        uk = invdx(xk[0], xk[1], xkp1[0], xkp1[1], stepSize)

        # Compute running cost for current step
        xkr = convert_x_y_to_radius(xkp1[0], xkp1[1])    
        stateCost = np.linalg.norm(xkr - xr + nanCorrector) * Qlist[k]
        controlCost = np.linalg.norm(uk - ur + nanCorrector) * Rlist[k]

        # Update the cost function
        cost += (stateCost + controlCost)

    return cost

# * SGD ------------------------------------------------------------------------

class SGD():
    def __init__(self, learningRate=1e-3, nEpoch=1000):
        self.__learningRate = learningRate
        self.__nEpoch = nEpoch
        self.__losses = onp.zeros(nEpoch)
        self.__gradients = None
        self.__iter = 0
    
    # * PUBLIC METHODS ---------------------------------------------------------

    def optimise_step(self, xr: float, ur: float):
        assert xr >= 0 and ur >= 0, "Negative target controls not permitted"
        
        self.__ur = ur**2
        self.__xr = xr

        self.__init_params__()

        loss_gradient = jax.grad(loss_function)

        for i in range(self.__nEpoch):
            self.__iter += 1

            loss = loss_function(
                    self.__X,
                    xr=self.__xr,
                    ur=self.__ur,
                    stepSize=self.__stepSize,
                    Qlist=self.__Qlist,
                    Rlist=self.__Rlist)

            self.__losses[i] = loss

            self.__gradients = loss_gradient(
                    self.__X,
                    xr=self.__xr,
                    ur=self.__ur,
                    stepSize=self.__stepSize,
                    Qlist=self.__Qlist,
                    Rlist=self.__Rlist)

            self.__update__()

            if self.__iter % 50 == 0:
                logger.info("{} epoch - Loss: {}".format(self.__iter, loss))


    def get_uncontrolled_trajectory_raw(self):
        xk = self.__x0[0]
        yk = self.__x0[1]

        # Create a control horizon x 2 matrix to store the positions
        trajectory = onp.zeros((self.__controlHorizon, 2))
        trajectory[0,0] = xk
        trajectory[0,1] = yk

        U = (onp.ones(self.__controlHorizon, dtype=float) * float(self.__ur))

        for k in range(1, len(U)):
            # Compute step
            derX = dx(x=xk, y=yk, uk=U[k])
            derY = dy(x=xk, y=yk, uk=U[k])

            xk += derX * self.__stepSize
            yk += derY * self.__stepSize

            # Save new coordinates to the trajectories matrix
            trajectory[k, 0] = xk
            trajectory[k, 1] = yk
            
        return trajectory

    # * PRIVATE METHODS --------------------------------------------------------

    def __update__(self):
        if self.__gradients == None:
            raise Exception("Gradient was not updated correctly")
            
        self.__X[1:,:] -= self.__gradients[1:,:] * self.__learningRate

    def __init_params__(self):
        # MPC specific stuff
        self.__controlHorizon = 20

        self.__x0 = [0.1, 0.1]

        self.__stepSize = 0.1

        # Get the uncontrolled trajectory
        self.__X = self.get_uncontrolled_trajectory_raw()

        # State weights
        self.__Qlist = onp.ones(self.__controlHorizon) * 3.0
        
        # Control weights
        self.__Rlist = onp.ones(self.__controlHorizon) * 0.1

    def get_losses(self):
        return self.__losses

    def get_controls(self):
        controls = [3.0]
        for k in range(1, len(self.__X)):
            xk = self.__X[k-1]
            xkp1 = self.__X[k]

            uk = invdx(xk[0], xk[1], xkp1[0], xkp1[1], self.__stepSize)
            
            controls.append(np.sqrt(uk))
        
        return controls

    def get_state_trajectory(self):
        trajectory = []

        for k in range(0, len(self.__X)):
            xk = self.__X[k]

            # Convert to r
            trajectory.append(convert_x_y_to_radius(xk[0], xk[1]))
        
        return trajectory

    def get_uncontrolled_trajectory(self):
        xk = self.__x0[0]
        yk = self.__x0[1]

        trajectory = []

        U = (onp.ones(self.__controlHorizon, dtype=float) * float(self.__ur))

        for k in range(0, len(U)):
            # Compute step
            derX = dx(x=xk, y=yk, uk=U[k])
            derY = dy(x=xk, y=yk, uk=U[k])

            xk += derX * self.__stepSize
            yk += derY * self.__stepSize

            # Convert to r
            trajectory.append(convert_x_y_to_radius(xk, yk))
        
        return trajectory

def run_SGD():
    lm = SGD(nEpoch=500, learningRate=0.01)
    xr = 3
    ur = 3
    lm.optimise_step(xr, ur)

    fig, axs = plt.subplots(3)

    fig.suptitle('MPC Toy problem - Collocation method - SGD')
    
    stateTrajectory = lm.get_state_trajectory()
    uncontrolled = lm.get_uncontrolled_trajectory()

    axs[0].plot(stateTrajectory, label="Optimal")
    axs[0].plot(uncontrolled, label="Uncontrolled")
    axs[0].plot(onp.arange(len(stateTrajectory)), 
            onp.ones(len(stateTrajectory))*xr, label="xr")
    axs[0].set(xlabel='k', ylabel="X")
    axs[0].legend()

    controls = lm.get_controls()
    axs[1].step(onp.arange(len(lm.get_controls())),lm.get_controls())
    axs[1].plot(onp.arange(len(controls)), 
            onp.ones(len(controls))*ur)
    axs[1].set(xlabel='k', ylabel="U")

    axs[2].plot(lm.get_losses())
    axs[2].set(xlabel='step', ylabel='loss')
    plt.show()

if __name__ == '__main__':
    run_SGD()