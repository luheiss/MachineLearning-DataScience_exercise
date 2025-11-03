import numpy as np
import matplotlib.pyplot as plt

# --- 1. Parameters ---
n_tosses = 100000  # Erhöht für bessere Darstellung des LLN
bias = 52  # Wahrscheinlichkeit für Heads ist 52% (p=0.52)
p_heads = bias / 100.0

# --- 2. Simulation Function (Vektorisiert) ---
def coin_toss(n_tosses, bias):
    random_numbers = np.random.randint(0, 100, size=n_tosses)   #1 Head - 0 Tail
    results = (random_numbers < bias).astype(int) # < bias -> 1 Head - 0 Tail and astype to int
    return results

# --- 3. Cumulative Averages Function --- 
def calculate_cumulative_averages(results):     #results is a numpy array of 0s and 1s
    cumulative_sums = np.cumsum(results)        #cumsum to get cumulative sum of heads
    toss_indices = np.arange(1, len(results) + 1)
    cumulative_averages = cumulative_sums / toss_indices
    return cumulative_averages

# --- 4. Execute Simulation ---
results_array = coin_toss(n_tosses, bias) 
heads_count = np.sum(results_array)
tails_count = n_tosses - heads_count

# Die Zählungen sind: 
# Number of heads: 52098
# Number of tails: 47902

# --- 5. Visualization of Final Counts (Pie Chart) ---
# ... (Code für Kuchendiagramm) ...

# --- 6. Calculate Cumulative Averages ---
cumulative_avg = calculate_cumulative_averages(results_array)

# --- 7. Visualization of LLN (Line Plot) ---
plt.figure(figsize=(10, 6))
toss_indices = np.arange(1, n_tosses + 1)

# Plot the simulation results
plt.plot(
    toss_indices,
    cumulative_avg,
    label='Beobachteter Anteil Heads',
    color='blue',
    alpha=0.8
)

# Plot the expected result line (the bias reference)
plt.axhline(
    y=p_heads,
    color='red',
    linestyle='--',
    linewidth=2,
    label=f'Erwartete Wahrscheinlichkeit ({p_heads:.2f})'
)

# Styling
plt.title(f'Voreingenommener Münzwurf (p={p_heads})', fontsize=16)
plt.xlabel('Anzahl der Münzwürfe (Log-Skala)', fontsize=14)
plt.ylabel('Anteil der Heads', fontsize=14)
plt.xscale('log') # Log-Skala, um die Variabilität am Anfang hervorzuheben
plt.legend(fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.tight_layout()
plt.show()