def make_full_graph(word_pos_num):
    full_g = dict()
    for word_pos_num_i in word_pos_num:
        for word_pos_num_j in word_pos_num:
            if word_pos_num_i[0] != word_pos_num_j[0] and word_pos_num_j[0] != 'root':

                if (word_pos_num_i[0], word_pos_num_i[1], word_pos_num_i[2]) in full_g:
                    (full_g[(word_pos_num_i[0], word_pos_num_i[1], word_pos_num_i[2])])[(word_pos_num_j[0], word_pos_num_j[1], word_pos_num_j[2])] = 0
                else:
                    full_g[(word_pos_num_i[0], word_pos_num_i[1], word_pos_num_i[2])] = \
                        {(word_pos_num_j[0], word_pos_num_j[1], word_pos_num_j[2]): 0}
    return full_g


def get_weighted_graph(g, features_list, w):
    for features_edge in features_list:
        weight = 0
        for feature_i in features_edge[2]:
            weight += w[feature_i]
        g[(features_edge[0][0], features_edge[0][1], features_edge[0][2])][(features_edge[1][0], features_edge[1][1], features_edge[1][2])] = -weight
    return g


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
# TODO: CHECK AGAIN!
def get_features_for_graph(features, g):
    features_list = []
    for head_data in g:
        for child_data in g[head_data]:
            dependency_arch = (head_data[0], head_data[1], child_data[0], child_data[1])
            features_list.append(((head_data[0], head_data[1], head_data[2]), (child_data[0], child_data[1], child_data[2]),
                                    get_features_for_arch(features, dependency_arch)))
    return features_list
