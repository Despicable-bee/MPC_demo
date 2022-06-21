# Standard libs

# Third party libs
import numpy as onp
from matplotlib import pyplot as plt


# local libs

# * Methods --------------------------------------------------------------------

def plot_vector_field(inf: float, sup: float, points: int):
    # Generate our grid of x and y points
    X,Y = onp.meshgrid(
            onp.linspace(inf, sup, points), 
            onp.linspace(inf, sup, points))
    
    # Transform cartesian to polar using parameterisation
    lamb = 9
    # R^2
    R2 = X**2 + Y**2
    xdot = (R2*lamb - R2**2 - Y*R2/X)/(X + Y**2/X)
    ydot = (R2 + Y*xdot) / X
    
    plt.quiver(X,Y,xdot,ydot)
    plt.xlim(inf, sup)
    plt.ylim(inf, sup)

def plot_trajectory(x0: float, y0: float, numSteps: int, stepSize: float):
    """ Plots a trajectory from a particular starting point """
    x_array = [x0]
    y_array = [y0]
    xi = x0
    yi = y0
    lamb = 9

    for t in range(numSteps):
        derX = dx(xi, yi, lamb)
        derY = dy(xi, yi, lamb)
        if derX == None or derY == None:
            break
        xi += derX * stepSize
        yi += derY * stepSize
        x_array.append(xi)
        y_array.append(yi)

    return x_array, y_array

def dx(x: float, y: float, lamb: float):
    try:
        r2 = x**2 + y**2
        xdot = (r2*lamb - r2**2 - y*r2/x)/(x + y**2/x)
   
    except Exception as e:
        print("{}, {}".format(x,y))
        return None
    return xdot

def dy(x: float, y: float, lamb: float):
    try:
        r2 = x**2 + y**2
        xdot = (r2*lamb - r2**2 - y*r2/x)/(x + y**2/x)
        ydot = (r2 + y*xdot) / x
    except Exception as e:
        print("{}, {}".format(x,y))
        return None
    return ydot

if __name__ == '__main__':
    

    points = []
    pointsTemp = onp.arange(-4,4,0.5)
    for i in pointsTemp:
        for j in pointsTemp:
            points.append((i, j))

    # Get the inside points
    pointsTemp = onp.arange(-0.1,0.1,0.05)
    # Get the outside points
    for i in pointsTemp:
        for j in pointsTemp:
            points.append((i, j))
    numSteps = 5000
    stepSize = 0.01

    plot_vector_field(inf=-4, sup=4, points=100)

    for point in points:
        xResult, yResult = plot_trajectory(point[0], point[1], numSteps, stepSize)
        plt.plot(xResult, yResult, label="{},{}".format(point[0], point[1]))
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()