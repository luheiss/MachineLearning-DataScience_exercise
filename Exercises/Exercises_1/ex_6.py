import numpy as np

N= 100000; #Number of Points, amount of random numbers
N_xy = []

def gen_random_N(N):
    for _ in range(N):
        N_xy.append(np.random.random(2))
    return np.array(N_xy)

list_Nxy= gen_random_N(N)
print(list_Nxy)

tran_list_Nxy = list_Nxy.T
x= tran_list_Nxy[0, :]
#print(x)
y= tran_list_Nxy[1, :]

#erg = x**2 + y**2
in_circle = x**2 + y**2 <= 1

M = np.sum(in_circle)

pi_estimate = 4 * (M / N)
print("Approximation of pi:", pi_estimate)