# Standard libs
from math import nan
from typing import List
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

#

logger = logging.getLogger('MPC Shooting method JAX')

# * SYSTEM FUNCTION ------------------------------------------------------------

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

def convert_x_y_to_radius(x: float, y: float):
    return np.sqrt(x**2 + y**2)

# * LOSS FUNCTION --------------------------------------------------------------

@jax.jit
def loss_function(params: List[float], xr: float, ur: float, x0: List[float], 
            u0: float, stepSize: float,
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
    xk = x0[0]
    yk = x0[1]

    # Compute initial state cost
    xkr = convert_x_y_to_radius(xk, yk)
    stateCost = np.linalg.norm(xkr - xr + nanCorrector) * Qlist[0]

    controlCost = np.linalg.norm(u0 - ur + nanCorrector) * Rlist[0]

    # Set initial cost
    cost = stateCost + controlCost

    for k in range(0, len(params)):
        # Compute step
        derX = dx(x=xk, y=yk, uk=params[k])
        derY = dy(x=xk, y=yk, uk=params[k])

        # Update the x and y positions
        xk += derX * stepSize
        yk += derY * stepSize

        # Compute running cost for current step
        xkr = convert_x_y_to_radius(xk, yk)    
        stateCost = np.linalg.norm(xkr - xr + nanCorrector) * Qlist[k + 1]
        controlCost = np.linalg.norm(params[k] - ur + nanCorrector) * Rlist[k + 1]

        # Update the cumulative cost function
        cost += (stateCost + controlCost)

    return cost

# * SGD ------------------------------------------------------------------------

class SGD():
    """
    
    """
    def __init__(self, learningRate=1e-3, nEpoch=1000):
        self.__learningRate = learningRate
        self.__nEpoch = nEpoch
        self.__losses = onp.zeros(nEpoch)
        self.__gradients = None
        self.__iter = 0

        
    
    def optimise_step(self, xr: float, ur: float):
        assert xr >= 0 and ur >= 0, "Negative target controls not permitted"
        
        self.__ur = ur**2
        self.__xr = xr

        self.__init__params__(ur)

        loss_gradient = jax.grad(loss_function)

        for i in range(self.__nEpoch):
            self.__iter += 1

            loss = loss_function(self.__U[1:], 
                    xr=self.__xr, 
                    ur=self.__ur, 
                    x0=self.__x0, 
                    u0=self.__U[0],
                    stepSize=self.__stepSize,
                    Qlist=self.__Qlist,
                    Rlist=self.__Rlist)

            # print(loss)
            # sys.exit()
            if math.isnan(loss):
                break

            self.__losses[i] = loss

            self.__gradients = loss_gradient(self.__U[1:], 
                    xr=self.__xr, 
                    ur=self.__ur, 
                    x0=self.__x0, 
                    u0=self.__U[0],
                    stepSize=self.__stepSize,
                    Qlist=self.__Qlist,
                    Rlist=self.__Rlist)
            #print(self.__gradients)

            self.__update__()
            
            #print(loss)
            
            if self.__iter % 50 == 0:
                # logging.info("Control inputs:")
                # print(self.__U)
                logger.info("{} epoch - Loss: {}".format(self.__iter, loss))

    def __init__params__(self, ur: float):
        # MPC specific stuff
        self.__controlHorizon = 20

        self.__x0 = [0.1, 0.1]
        self.__U = (onp.ones(self.__controlHorizon, dtype=float) * float(ur)**2)

        #print(self.__U)

        # State weights
        self.__Qlist = onp.ones(self.__controlHorizon) * 3.0
        
        # Control weights
        self.__Rlist = onp.ones(self.__controlHorizon) * 0.1

        self.__stepSize = 0.1
    
    def __update__(self):
        self.__U[1:] -= self.__gradients * self.__learningRate

    def get_losses(self):
        return self.__losses

    def get_controls(self):
        return np.sqrt(self.__U)

    def get_state_trajectory(self):
        xk = self.__x0[0]
        yk = self.__x0[1]

        trajectory = []

        for k in range(0, len(self.__U)):
            # Compute step
            derX = dx(x=xk, y=yk, uk=self.__U[k])
            derY = dy(x=xk, y=yk, uk=self.__U[k])

            xk += derX * self.__stepSize
            yk += derY * self.__stepSize

            # Convert to r
            trajectory.append(convert_x_y_to_radius(xk, yk))
        
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
    lm = SGD(nEpoch=850, learningRate=0.01)
    xr = 3
    ur = 3
    lm.optimise_step(xr, ur)

    fig, axs = plt.subplots(3)

    fig.suptitle('MPC Toy problem - Shooting method - SGD')
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