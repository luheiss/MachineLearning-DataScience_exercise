import numpy as np

vec = np.array([5.0, -12.0, 0.0])

mat = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

a = np.random.rand(3)
mat_A = np.outer(a, a) + np.eye(3)

print(f"Vektor: {vec}")
print(f"Matrix B:\n{mat}")
print(f"Matrix A (Pos. Def.):\n{mat_A}")
print("-" * 20)


print("--- Normen ---")
print(f"2-Norm (Euklidisch) vec: {np.linalg.norm(vec, ord=2):.2f}")
print(f"1-Norm vec: {np.linalg.norm(vec, ord=1):.2f}")
print(f"Infinity-Norm vec: {np.linalg.norm(vec, ord=np.inf):.2f}")
print(f"Frobenius-Norm mat: {np.linalg.norm(mat, ord='fro'):.2f}")
print("-" * 20)

print("--- Statistik ---")
print(f"Median mat: {np.median(mat):.1f}")
print(f"Mean (Gesamt) mat: {np.mean(mat):.1f}")
print(f"Mean (Achse 0/Spalten): {np.mean(mat, axis=0)}")
print(f"Variance mat: {np.var(mat):.2f}")
print(f"Standard Deviation mat: {np.std(mat):.2f}")
print("-" * 20)


data_flat = mat.flatten()
weights = np.array([1, 1, 1, 1, 1, 1, 5, 5, 5])
avg_weighted = np.average(data_flat, weights=weights)

print("--- Average mit Gewichten ---")
print(f"Weighted Average: {avg_weighted:.2f}")
print("-" * 20)

print("--- Varianz Manuell (Vektor) ---")
mean_vec = np.mean(vec)
variance_manual = np.mean((vec - mean_vec)**2) 

print(f"Mean: {mean_vec:.2f}")
print(f"Manuelle Varianz: {variance_manual:.2f}")
print("-" * 20)

L = np.linalg.cholesky(mat_A)

print("--- Cholesky-Zerlegung (L) ---")
print(L)