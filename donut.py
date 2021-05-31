#!/usr/bin/python3

import numpy as np
import time

def generate_points():
	# Create the points of the "main" circle. They are going
	# to be repeated (0,0,0,1,1,1,...) , so when we multiply
	# by the rotation matrix Ry, we obtain all the points:
	circle_points = np.repeat(
		np.transpose(
			np.array(
				[[R2 + R1*np.cos(theta)],
				 [R1*np.sin(theta)],
				 [np.zeros(N_theta)]], float),
			axes = (2, 0, 1)),
		N_phi, axis = 0)

	normal_circle_points = np.repeat(
		np.transpose(
			np.array(
				[[np.cos(theta)],
				 [np.sin(theta)],
				 [np.zeros(N_theta)]], float),
			axes = (2, 0, 1)),
		N_phi, axis = 0)

	# Rotation matrix. It is tiled (0,1,2,0,1,2...), bc
	# we have to match the dimensions of the circle_points:
	Ry = np.tile(
		np.transpose(
			np.array(
				[[ np.cos(phi)    , np.zeros(N_phi), np.sin(phi)],
				 [ np.zeros(N_phi), np.ones(N_phi) , np.zeros(N_phi)],
				 [-np.sin(phi)    , np.zeros(N_phi), np.cos(phi)]], float),
			axes = (2, 0, 1)),
		(N_theta, 1, 1))

	# Main points of the toroid at the beginning. We obtain
	# them applying (multiplying by) the rotation matrix
	# to the points of the main circle.
	points = np.matmul(Ry, circle_points)
	# and the normal points:
	normal_points = np.matmul(Ry, normal_circle_points)
	return points, normal_points

def compute_output():
	# Compute the corresponding x,y pixel points projected on screen:
	x = points[:,0,:].flatten()
	y = points[:,1,:].flatten()
	z = points[:,2,:].flatten() + K2 # add the offset
	screen_points = np.transpose(
		np.array(
			[screen_width/2 + K1 * x / z,
			 screen_height/2 - K1 * y / z], int))

	# test if faster with variable assignation or directly without assigning variables
	# faster without assigning them (100 microseconds or so)

	# Create a new matrix to store the indices of the corresponding points for
	# each pixel, to manipulate them afterwards:
	index_matrix = np.zeros((screen_height, screen_width)).tolist()
	i = int(0)
	for column, row in screen_points: # columns: x; rows: y
		if index_matrix[row][column]:
			index_matrix[row][column].append(i)
		else:
			index_matrix[row][column] = [i]
		i += 1

	# Iterate over each pixel, getting the closest point to viewer for each of them,
	# and compute the luminance for that point. Afterwards, assign the corresponding
	# character to the output matrix to display.
	luminance_direction = np.array([0, 1/np.sqrt(2), 1/np.sqrt(2)], float)
	output = np.full((screen_height, screen_width), ' ')
	i, j = int(0), int(0)
	for pixel_row in index_matrix:
		for pixel in pixel_row:
			# pixel is a list with the indices of the points corresponding to that pixel.
			# First check if it's an empty pixel:
			if pixel:
				# We have to get the point with minimum z value from the ones pointed by the indices in pixel
				# pixel_points = points
				index = np.argmin(
						points[pixel], # get the points selected by the pixel's indices
						axis = 0       # get the index for the minimum value for each coordinate
						)[2,0]         # and select the z value
				point = points[index]  # and use it as an index to assign the point to the pixel
				normal = normal_points[index]
				luminance_index = int(np.matmul(luminance_direction, normal)[0] * 11)
				if luminance_index >= 0:
					output[i,j] = ".,-~:;=!*#$@"[luminance_index]
				else:
					output[i,j] = "."
			j += 1
		i += 1
		j = 0
	return output

def dump_on_screen():
	print("\x1b[H", end='')
	for i in range(screen_height):
		for j in range(screen_width):
			print(output[i,j], end='')
		print()

def rotate():
	# Generate both rotation matrices:
	Rx = np.array(
		[[1, 0        ,  0        ],
		 [0, np.cos(A), -np.sin(A)],
		 [0, np.sin(A),  np.cos(A)]], float)
	Rz = np.array(
		[[np.cos(B), -np.sin(B), 0],
		 [np.sin(B),  np.cos(B), 0],
		 [0        , 0         , 1]], float)
	# and apply the rotations:
	rotated_points = np.matmul(Rz, np.matmul(Rx, points))
	rotated_normal_points = np.matmul(Rz, np.matmul(Rx, normal_points))
	return rotated_points, rotated_normal_points


# to avoid division by 0 error messages:
np.seterr(divide='ignore', invalid='ignore')

screen_width = 80
screen_height = 40

N_theta = 100
theta = np.linspace(0, 2*np.pi, N_theta)
N_phi = 300
phi = np.linspace(0, 2*np.pi, N_phi)
R1 = 1
R2 = 2
K1 = 60
K2 = 10

# rotation angles:
A = 0.1
B = 0.1

points, normal_points = generate_points()
t_previous_frame = time.time()
while True:
	output = compute_output()
	while (time.time() - t_previous_frame) < 0.083:
		time.sleep(0.005)
	dump_on_screen()
	t_previous_frame = time.time()
	points, normal_points = rotate()
