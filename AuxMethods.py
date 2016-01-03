# Methods that both the trainer and the inference process use
import networkx as nx


# Get (word,pos) data and make a full graph
def build_full_graph(words_pos):
    full_g = nx.DiGraph()
    for word_pos_i in words_pos:
        for word_pos_j in words_pos:
            if word_pos_i[0] != word_pos_j[0] and word_pos_j[0] != 'root':
                full_g.add_edge((word_pos_i[0], word_pos_i[1]), (word_pos_j[0], word_pos_j[1]), weight=0)
    return full_g


# Return the features number that return 1 for a given arch
def get_features_for_arch(features, dependency_arch):
    num_features = []
    if ((dependency_arch[0], dependency_arch[1]), 1) in features:
        num_features.append(features[((dependency_arch[0], dependency_arch[1]), 1)])
    if ((dependency_arch[0], ''), 2) in features:
        num_features.append(features[((dependency_arch[0], ''), 2)])
    if ((dependency_arch[1], ''), 3) in features:
        num_features.append(features[((dependency_arch[1], ''), 3)])
    if ((dependency_arch[2], dependency_arch[3]), 4) in features:
        num_features.append(features[((dependency_arch[2], dependency_arch[3]), 4)])
    if ((dependency_arch[2], ''), 5) in features:
        num_features.append(features[((dependency_arch[2], ''), 5)])
    if ((dependency_arch[3], ''), 6) in features:
        num_features.append(features[((dependency_arch[3], ''), 6)])
    if ((dependency_arch[1], (dependency_arch[2], dependency_arch[3])), 8) in features:
        num_features.append(features[((dependency_arch[1], (dependency_arch[2], dependency_arch[3])), 8)])
    if (((dependency_arch[0], dependency_arch[1]), dependency_arch[3]), 10) in features:
        num_features.append(features[(((dependency_arch[0], dependency_arch[1]), dependency_arch[3]), 10)])
    if ((dependency_arch[1], dependency_arch[3]), 13) in features:
        num_features.append(features[((dependency_arch[1], dependency_arch[3]), 13)])
    return num_features


# Get a graph, and return a list of lists
# Each list holds the features number that return 1 for that arch
def get_features_for_graph(g, features):
    features_list = []
    for edge in g.edges(data=True):
        dependency_arch = (edge[0][0], edge[0][1], edge[1][0], edge[1][1])
        features_list.append((edge[0], edge[1], get_features_for_arch(features, dependency_arch)))
    return features_list


# Get a graph and a features list that fit to his arches
# Return a weighted graph (according to the features)
def get_weighted_graph(g, features_list, w):
    for features_edge in features_list:
        weight = 0
        for feature_i in features_edge[2]:
            weight += w[feature_i]
        g[features_edge[0]][features_edge[1]]['weight'] = -weight
    return g


# Because we send to the MST an undirected graph - we need to find the directed
def get_directed_graph(g_undirected, g_direct):
    E = set(g_undirected.edges())
    new_edges = [e for e in g_direct.edges() if e in E or reversed(e) in E]
    return nx.DiGraph(new_edges)
