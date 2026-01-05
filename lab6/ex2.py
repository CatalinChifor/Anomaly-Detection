import networkx as nx
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import random

def get_oddball_scores(G, x_attr, y_attr):
    nodes = list(G.nodes())
    features = []
    for n in nodes:
        ego = nx.ego_graph(G, n)
        ni = len(ego.nodes()) - 1
        ei = ego.size()
        wi = ego.size(weight='weight')
        features.append({'Ni': ni, 'Ei': ei, 'Wi': wi})
    
    X_val = np.array([f[x_attr] for f in features]).reshape(-1, 1)
    y_val = np.array([f[y_attr] for f in features])
    
    log_X, log_y = np.log(X_val + 1), np.log(y_val + 1)
    model = LinearRegression().fit(log_X, log_y)
    y_pred = np.exp(model.predict(log_X))
    
    scores = []
    for i, n in enumerate(nodes):
        yi, cxi = y_val[i], y_pred[i]
        s = (max(yi, cxi) / max(min(yi, cxi), 1e-6)) * np.log(abs(yi - cxi) + 1)
        scores.append((n, s))
    return scores

G1 = nx.random_regular_graph(3, 100)
G2 = nx.connected_caveman_graph(10, 20)
G_merged = nx.disjoint_union(G1, G2)

nodes = list(G_merged.nodes())
for _ in range(10):
    G_merged.add_edge(random.choice(nodes), random.choice(nodes))

scores_clique = get_oddball_scores(G_merged, 'Ni', 'Ei')
scores_clique.sort(key=lambda x: x[1], reverse=True)
top_10_cliques = [s[0] for s in scores_clique[:10]]

H1 = nx.random_regular_graph(3, 100)
H2 = nx.random_regular_graph(5, 100)
G_heavy = nx.disjoint_union(H1, H2)

nx.set_edge_attributes(G_heavy, 1, 'weight')

random_nodes = random.sample(list(G_heavy.nodes()), 2)
for root in random_nodes:
    ego_edges = nx.ego_graph(G_heavy, root).edges()
    for u, v in ego_edges:
        G_heavy[u][v]['weight'] += 10

scores_heavy = get_oddball_scores(G_heavy, 'Ei', 'Wi')
scores_heavy.sort(key=lambda x: x[1], reverse=True)
top_4_heavy = [s[0] for s in scores_heavy[:4]]

plt.figure(figsize=(8, 6))
colors = ['red' if n in top_4_heavy else 'skyblue' for n in G_heavy.nodes()]
nx.draw(G_heavy, node_color=colors, with_labels=False, node_size=50)
plt.title("Heavy Vicinity Detection (Top 4 in Red)")
plt.show()