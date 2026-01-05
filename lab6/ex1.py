import networkx as nx
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

G = nx.Graph()
with open('ca-AstroPh.txt', 'r') as f:
    for i, line in enumerate(f):
        if i >= 1500 or line.startswith('#'): continue
        u, v = line.strip().split()
        if G.has_edge(u, v):
            G[u][v]['weight'] += 1
        else:
            G.add_edge(u, v, weight=1)

data = []
nodes = list(G.nodes())
for n in nodes:
    ego = nx.ego_graph(G, n)
    ni = len(ego.nodes()) - 1
    ei = ego.size()
    data.append([ni, ei])
    nx.set_node_attributes(G, {n: {"Ni": ni, "Ei": ei}})

X = np.log(np.array([d[0] for d in data]).reshape(-1, 1) + 1)
y = np.log(np.array([d[1] for d in data]) + 1)

model = LinearRegression().fit(X, y)
y_pred = np.exp(model.predict(X))
y_actual = np.exp(y)

scores = []
for i, n in enumerate(nodes):
    yi, cxi = y_actual[i], y_pred[i]
    s = (max(yi, cxi) / max(min(yi, cxi), 1e-6)) * np.log(abs(yi - cxi) + 1)
    scores.append((n, s))

scores.sort(key=lambda x: x[1], reverse=True)
top_10 = [s[0] for s in scores[:10]]
colors = ['red' if n in top_10 else 'skyblue' for n in G.nodes()]

nx.draw(G, node_color=colors, with_labels=False, node_size=30)
plt.show()