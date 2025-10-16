import numpy as np
import matplotlib.pyplot as plt

v1 = [1.25,4.1]
v2 = [5,7.4]
z1 = [2.6,4.25]
z2 = [2,7]

a =[1.4,5.1]
b =[4.7,7.0]

dirv_v1 = np.array(v2) - np.array(v1)
dirv_z1 = np.array(z2) - np.array(z1)

ortho_v1 = np.array([dirv_v1[1], -dirv_v1[0]])
ortho_z1 = np.array([dirv_z1[1], -dirv_z1[0]])

# Überprüfung der Orthogonalität
print("check", np.dot(ortho_z1, dirv_z1))
print("w", ortho_v1)

bv = np.dot(ortho_v1, v1)
bz = np.dot(ortho_z1, z1)

#Klassifikation
#< > das bedeutet skalarprodukt
#<w1,a> +b


plt.figure(figsize=(8, 6))
plt.scatter(*v1, color='blue', label='v1 (1.25, 4.1)', s=100)
plt.scatter(*v2, color='blue', label='v2 (5, 7.4)', s=100)
plt.scatter(*z1, color='red', label='z1 (2.6, 4.25)', s=100)
plt.scatter(*z2, color='red', label='z2 (2, 7)', s=100)
plt.scatter(*a, color='green', label='a (1.4, 5.1)', s=100)
plt.scatter(*b, color='green', label='b (4.7, 7.0)', s=100)

plt.show()