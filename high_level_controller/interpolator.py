# This file will be responsible for taking an action chunk with target times for the robot's position
# and interpolating the robot's position at each time step. 

# It should support the following interpolation methods:
# - Linear interpolation
# - Cubic spline interpolation
# - Quintic spline interpolation

# The interpolation should set target positions and orientations for the robot's end effector:
# - Position: (x, y, z)
# - Orientation: (roll, pitch, yaw)

# the actions that the model will send the mid level interpolator will roughly be of the form:
# array of X,Y,Z, r, p, y, time

