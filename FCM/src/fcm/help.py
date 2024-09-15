import numpy as np
import pandas as pd
import skfuzzy as fuzz
import matplotlib.pyplot as plt

df = pd.read_csv('Country-data.csv')

features = ['exports', 'imports']
X = df[features].values.T 

n_clusters = 3
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(X, c=n_clusters, m=2, error=0.005, maxiter=1000)

cluster_membership = np.argmax(u, axis=0)

max_u = np.max(u, axis=0)

plt.figure(figsize=(10, 7))

scatter = plt.scatter(X[0], X[1], c=max_u, cmap='plasma', s=100, alpha=0.6)

plt.scatter(cntr[:, 0], cntr[:, 1], color='red', marker='x', s=150, label="Centers")

cbar = plt.colorbar(scatter)
cbar.set_label('Degree of Membership')

plt.title("Fuzzy C-Means Clustering with Dataset HELP")
plt.xlabel("Exports")
plt.ylabel("Imports")
plt.legend()
plt.show()
