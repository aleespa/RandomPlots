import matplotlib.pylab as plt
import numpy as np

# colors= ['#e74c3c','#f1c40f' ,'#2ecc71','#e67e22','#27ae60']
# p = plt.figure(figsize=(12,12),facecolor='black',dpi=400)
# p = plt.axis('off')
# p = plt.xlim(1,2)
# p = plt.ylim(0,1)
# for i in np.linspace(0.1,5,120):
#     p = plt.plot(np.linspace(1,2,100),[x**i for x in np.linspace(0,1,100)],alpha=np.random.beta(1,1),lw = np.random.choice([7,10,3,2,5]),color = np.random.choice(colors))
#
colors = ['#e74c3c', '#f1c40f', '#2ecc71', '#e67e22', '#27ae60']
p = plt.figure(figsize=(16, 9), facecolor='black', dpi=400)
p = plt.axis('off')
p = plt.xlim(1.3, 1.8)
p = plt.ylim(0, 0.5)
for i in np.linspace(0.1, 5, 150):
    p = plt.plot(
        np.linspace(1, 2, 100),
        [x**i for x in np.linspace(0, 1, 100)],
        alpha=np.random.beta(1, 1),
        lw=np.random.choice([1, 5, 3, 2, 5]),
        color=np.random.choice(colors),
    )

p = plt.savefig(f'C:/Users/Alejandro/Pictures/RandomPlots/wall.png', facecolor='black')
