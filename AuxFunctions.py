import BasicFunctions
import ImprovedFunctions


# Get a list of (word,pod,word num) and build a valid full graph
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


# Get a graph, list of features that apply on him and vector w - return the weighted graph
def get_weighted_graph(g, features_list, w):
    for features_edge in features_list:
        weight = 0
        for feature_i in features_edge[2]:
            weight += w[feature_i]
        g[(features_edge[0][0], features_edge[0][1], features_edge[0][2])][(features_edge[1][0], features_edge[1][1], features_edge[1][2])] = -weight
    return g


# Get a graph, and return a list of lists
# Each list holds the features number that return 1 for that arch (and the arch data)
def get_features_for_graph(features, g, g_words_tags, is_improved):
    features_list = []
    for head_data in g:
        for child_data in g[head_data]:
            dependency_arch = (head_data[0], head_data[1], child_data[0], child_data[1])
            if is_improved:
                features_list.append(((head_data[0], head_data[1], head_data[2]), (child_data[0], child_data[1], child_data[2]),
                                    ImprovedFunctions.get_features_for_arch(features, dependency_arch, g_words_tags,
                                                                            child_data[2], head_data[2])))
            else:
                features_list.append(((head_data[0], head_data[1], head_data[2]), (child_data[0], child_data[1], child_data[2]),
                                    BasicFunctions.get_features_for_arch(features, dependency_arch)))
    return features_list
