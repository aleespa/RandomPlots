import matplotlib.pylab as plt
import numpy as np
import seaborn as sns

p = plt.figure(figsize=(12, 12), facecolor='black', dpi=400)
p = plt.axis('off')

for m in np.linspace(-10, 10, 20):
    x = np.random.normal(m, 1, size=15)
    p = sns.kdeplot(x, shade=True)
p = plt.ylim(-0.60, 1)
plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/07012020.png', facecolor='black')
