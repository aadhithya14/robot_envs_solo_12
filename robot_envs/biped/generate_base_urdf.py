import numpy as np

box_size = [0.12, 0.14, 0.07]
mass = 0.8

#box_size = [0.39, 0.17, 0.05]
#mass = 1.3

inertia_matrix = np.zeros([3, 3])
inertia_matrix[0, 0] = 1.0 / 12.0 * mass * (box_size[1] ** 2 + box_size[2] ** 2)
inertia_matrix[1, 1] = 1.0 / 12.0 * mass * (box_size[0] ** 2 + box_size[2] ** 2)
inertia_matrix[2, 2] = 1.0 / 12.0 * mass * (box_size[0] ** 2 + box_size[1] ** 2)

print(inertia_matrix)
