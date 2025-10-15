import random
import math #omport accurate value of pi

N= 1_000_000; #Number of Points, amount of random numbers

# Generate a random float between 0 and 1
x = [random.uniform(0, 1) for _ in range(N)]
#print(x)
y = [random.uniform(0, 1) for _ in range(N)]
#print(y)

# calculate M
def is_in_circle(x, y):
    return x**2 + y**2 <= 1

M = sum(is_in_circle(x[i], y[i]) for i in range(N))
print(M)

# estimate pi
def estimate_pi(M, N):
    return 4 * (M / N)
estimate_pi = estimate_pi(M, N)
print("Approximation of pi:", estimate_pi)

def get_accuracy(N):
    diff = math.pi - estimate_pi
    return diff

accuracy = get_accuracy(N)
print("Accuracy:", accuracy)