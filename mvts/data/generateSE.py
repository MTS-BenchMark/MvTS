from mvts.data.node2vec import Graph
import numpy as np
import networkx as nx
from gensim.models import Word2Vec

is_directed = True
p = 2
q = 1
num_walks = 100
walk_length = 80
dimensions = 64
window_size = 10
iter = 1000


def read_graph(edgelist):
    G = nx.read_edgelist(
        edgelist, nodetype=int, data=(('weight', float),),
        create_using=nx.DiGraph())

    return G


def learn_embeddings(walks, dimensions, output_file):
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, vector_size=dimensions, window=10, min_count=0, sg=1,
                     workers=8, epochs=iter)
    model.wv.save_word2vec_format(output_file)
    return

def generateSE(Adj_file, SE_file):
    nx_G = read_graph(Adj_file)
    G = Graph(nx_G, is_directed, p, q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks, walk_length)
    learn_embeddings(walks, dimensions, SE_file)
