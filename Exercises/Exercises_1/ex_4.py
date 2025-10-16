import matplotlib.pyplot as plt
from math import pi, sqrt, exp

def f(x, mu, sigma):
    faktor = 1 / (sqrt(2 * pi * sigma**2))
    exponent = -((x - mu)**2) / (2 * sigma**2)
    return faktor * exp(exponent)

#Chat
def create_data(mu, sigma, start=-5.0, end=5.0, step=0.1):
    xs = []
    ys = []

    i = int(start / step)
    while (i * step) <= end + step/2: 
        x = i * step
        xs.append(x)
        ys.append(f(x, mu, sigma))
        i += 1
    return xs, ys
#

params = [
    {'mu': 1, 'sigma': 5, 'color': 'blue'},
    {'mu': 0, 'sigma': 1, 'color': 'red'},
    {'mu': -2, 'sigma': 0.5, 'color': 'green'}
]

plt.figure(figsize=(10, 6))

for p in params:
    mu = p['mu']
    sigma = p['sigma']
    
    xs, ys = create_data(mu, sigma)
    
    # Plot mit Legende
    plt.plot(xs, ys, 
             color=p['color'],
             label=f'μ = {mu}, σ = {sigma}')

# Diagramm-Anpassungen
plt.title('Visualisierung der Normalverteilung (Gaußkurve)')
plt.xlabel('x')
plt.ylabel('Wahrscheinlichkeitsdichte f(x)')
plt.legend()
plt.grid(True)
plt.show()