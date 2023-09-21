from pathlib import Path
from torch_geometric.datasets import Planetoid
from sklearn.decomposition import PCA
import scipy.sparse as ss
import numpy as np
from dataloader import SparseGraph, save_sparse_graph_to_npz


path = Path('data/planetoid')
dataset = Planetoid(path, 'cora')
data = dataset[0]

pca = PCA(n_components=128)
reduced_features = pca.fit_transform(data.x)

print("Original data shape: ", data.x.shape)
print("Reduced data shape: ", reduced_features.shape)

edge_list = data.edge_index.numpy()
ones = np.ones(data.num_edges)
adj_matrix = ss.csr_matrix((ones, (edge_list[0], edge_list[1])))

sparse_graph = SparseGraph(adj_matrix, attr_matrix=reduced_features, labels=data.y.numpy())
save_sparse_graph_to_npz(f'data/cora.npz', sparse_graph)
