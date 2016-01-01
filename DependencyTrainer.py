from datetime import datetime
import networkx as nx
import numpy as np
import pickle


class DependencyTrainer:
    FEATURE_1_LIMIT = 2
    FEATURE_2_LIMIT = 2
    FEATURE_3_LIMIT = 0
    FEATURE_4_LIMIT = 5
    FEATURE_5_LIMIT = 5
    FEATURE_6_LIMIT = 0
    FEATURE_8_LIMIT = 10
    FEATURE_10_LIMIT = 10
    FEATURE_13_LIMIT = 0

    def __init__(self):
        self.feature_1_dict = dict()
        self.feature_2_dict = dict()
        self.feature_3_dict = dict()
        self.feature_4_dict = dict()
        self.feature_5_dict = dict()
        self.feature_6_dict = dict()
        self.feature_8_dict = dict()
        self.feature_10_dict = dict()
        self.feature_13_dict = dict()

        self.arches_data_for_graphs = []

        self.features = dict()
        self.feature_num = 0

        self.graphs = []  # Holds (graph, full_graph)
        self.scored_graphs = []  # Holds (graph, graph features,  full_graph, full_graph features)

        self.f_vector_results = dict()

    #################
    # FEATURES PART #
    #################

    # Add all seen features to the dicts
    def get_features(self):
        sentence_words_pos = dict()
        with open('Data\\train.labeled', 'r') as f:
            for line in f:
                if line is '\n':
                    arches = set()
                    data_for_full_graph = set()
                    for counter in sentence_words_pos:
                        word_tuple = sentence_words_pos[counter]
                        # dependency_arch= (head, head_pos, child, child_pos)
                        if word_tuple[2] == 0:
                            dependency_arch = ('root', 'root', word_tuple[0], word_tuple[1])
                        else:
                            dependency_arch = (sentence_words_pos[word_tuple[2]][0],
                                               sentence_words_pos[word_tuple[2]][1], word_tuple[0], word_tuple[1])
                        self.add_features_to_dicts(dependency_arch)

                        arches.add(dependency_arch)
                        data_for_full_graph.add((dependency_arch[0], dependency_arch[1]))
                        data_for_full_graph.add((dependency_arch[2], dependency_arch[3]))
                    self.arches_data_for_graphs.append((arches, list(data_for_full_graph)))

                    sentence_words_pos = dict()
                else:
                    split_line = line.split('\t')
                    # (counter)->(token,pos,head)
                    sentence_words_pos[int(split_line[0])] = (split_line[1], split_line[3], int(split_line[6]))
        f.close()

    # Get a dependency arch, and add all the possible features to the dict
    def add_features_to_dicts(self, dependency_arch):
        self.add_feature_for_dependency((dependency_arch[0], dependency_arch[1]), self.feature_1_dict)
        self.add_feature_for_dependency((dependency_arch[0], ''), self.feature_2_dict)
        self.add_feature_for_dependency((dependency_arch[1], ''), self.feature_3_dict)
        self.add_feature_for_dependency((dependency_arch[2], dependency_arch[3]), self.feature_4_dict)
        self.add_feature_for_dependency((dependency_arch[2], ''), self.feature_5_dict)
        self.add_feature_for_dependency((dependency_arch[3], ''), self.feature_6_dict)
        self.add_feature_for_dependency((dependency_arch[1], (dependency_arch[2], dependency_arch[3])),
                                        self.feature_8_dict)
        self.add_feature_for_dependency(((dependency_arch[0], dependency_arch[1]), dependency_arch[3]),
                                        self.feature_10_dict)
        self.add_feature_for_dependency((dependency_arch[1], dependency_arch[3]), self.feature_13_dict)

    # Add a specific feature (to a specific feature dict)
    @staticmethod
    def add_feature_for_dependency(feature_data, feature_dict):
        if feature_data in feature_dict:
            feature_dict[feature_data] += 1
        else:
            feature_dict[feature_data] = 1

    ################
    # BUILD GRAPHS #
    ################

    # Create a graph representation of the sentence (for edmond)
    # Create a full graph representation of the sentence (for edmond)
    def build_graphs(self):
        for arches_data_list_data in self.arches_data_for_graphs:
            g = nx.DiGraph()
            for arch in arches_data_list_data[0]:
                g.add_edge((arch[0], arch[1]), (arch[2], arch[3]), weight=0)
            self.graphs.append((g, self.build_full_graph(arches_data_list_data[1])))

    @staticmethod
    # Get (word,pos) data and make a full graph
    def build_full_graph(words_pos):
        full_g = nx.DiGraph()
        for word_pos_i in words_pos:
            for word_pos_j in words_pos:
                if word_pos_i[0] != word_pos_j[0] and word_pos_j[0] != 'root':
                    full_g.add_edge((word_pos_i[0], word_pos_i[1]), (word_pos_j[0], word_pos_j[1]), weight=0)
        return full_g

    ##########################
    # FREQUENT FEATURES PART #
    ##########################

    # Get only frequent features (configurable limits)
    def get_frequent_features(self):
        self.add_frequent_feature(self.FEATURE_1_LIMIT, self.feature_1_dict, 1)
        self.add_frequent_feature(self.FEATURE_2_LIMIT, self.feature_2_dict, 2)
        self.add_frequent_feature(self.FEATURE_3_LIMIT, self.feature_3_dict, 3)
        self.add_frequent_feature(self.FEATURE_4_LIMIT, self.feature_4_dict, 4)
        self.add_frequent_feature(self.FEATURE_5_LIMIT, self.feature_5_dict, 5)
        self.add_frequent_feature(self.FEATURE_6_LIMIT, self.feature_6_dict, 6)
        self.add_frequent_feature(self.FEATURE_8_LIMIT, self.feature_8_dict, 8)
        self.add_frequent_feature(self.FEATURE_10_LIMIT, self.feature_10_dict, 10)
        self.add_frequent_feature(self.FEATURE_13_LIMIT, self.feature_13_dict, 13)

    # For a specific feature dict
    def add_frequent_feature(self, limit, feature_dict, dict_num):
        counter = 0
        for key in feature_dict:
            if feature_dict[key] > limit:
                if key not in self.features:
                    self.features[(key, dict_num)] = self.feature_num
                    self.feature_num += 1
                    counter += 1
        print(str(counter) + ' of feature ' + str(dict_num))

    #################
    # Pre Calculate #
    #################

    # Return the features number that return 1 for a given arch
    def get_features_for_arch(self, dependency_arch):
        num_features = []
        if ((dependency_arch[0], dependency_arch[1]), 1) in self.features:
            num_features.append(self.features[((dependency_arch[0], dependency_arch[1]), 1)])
        if ((dependency_arch[0], ''), 2) in self.features:
            num_features.append(self.features[((dependency_arch[0], ''), 2)])
        if ((dependency_arch[1], ''), 3) in self.features:
            num_features.append(self.features[((dependency_arch[1], ''), 3)])
        if ((dependency_arch[2], dependency_arch[3]), 4) in self.features:
            num_features.append(self.features[((dependency_arch[2], dependency_arch[3]), 4)])
        if ((dependency_arch[2], ''), 5) in self.features:
            num_features.append(self.features[((dependency_arch[2], ''), 5)])
        if ((dependency_arch[3], ''), 6) in self.features:
            num_features.append(self.features[((dependency_arch[3], ''), 6)])
        if ((dependency_arch[1], (dependency_arch[2], dependency_arch[3])), 8) in self.features:
            num_features.append(self.features[((dependency_arch[1], (dependency_arch[2], dependency_arch[3])), 8)])
        if (((dependency_arch[0], dependency_arch[1]), dependency_arch[3]), 10) in self.features:
            num_features.append(self.features[(((dependency_arch[0], dependency_arch[1]), dependency_arch[3]), 10)])
        if ((dependency_arch[1], dependency_arch[3]), 13) in self.features:
            num_features.append(self.features[((dependency_arch[1], dependency_arch[3]), 13)])
        return num_features

    # Get a graph, and return a list of lists
    # Each list holds the features number that return 1 for that arch
    def get_features_for_graph(self, g):
        features_list = []
        for edge in g.edges(data=True):
            dependency_arch = (edge[0][0], edge[0][1], edge[1][0], edge[1][1])
            features_list.append((edge[0], edge[1], self.get_features_for_arch(dependency_arch)))
        return features_list

    # Doing this calculation one time !
    # checking the features for each graph (real and full)
    def calculate_features_for_all_graphs(self):
        for data in self.graphs:
            self.scored_graphs.append((data[0], self.get_features_for_graph(data[0]),
                                       data[1], self.get_features_for_graph(data[1])))

    ########################
    # Perceptron Algorithm #
    ########################

    @staticmethod
    # Get a graph and a features list that fit to his arches
    # Return a weighted graph (according to the features)
    def get_weighted_graph(g, features_list, w):
        for features_edge in features_list:
            weight = 0
            for feature_i in features_edge[2]:
                weight += w[feature_i]
            g[features_edge[0]][features_edge[1]]['weight'] = -weight
        return g

    # Get a graph and return "f_vector" for him
    def get_f_vector(self, g):
        f_vector = np.zeros(self.feature_num, dtype=int)
        features = self.get_features_for_graph(g)
        for feature_data in features:
            for feature_i in feature_data[2]:
                f_vector[feature_i] += 1
        return f_vector

    # If the f_vector of a real graph is already calculated - return the result
    # Else, calculate the features and save to the dictionary
    def get_f_vector_real(self, g):
        if g in self.f_vector_results:
            return self.f_vector_results[g]
        else:
            temp = self.get_f_vector(g)
            self.f_vector_results[g] = temp
            return temp

    # Because we send to the MST an undirected graph - we need to find the directed
    @staticmethod
    def get_directed_graph(g_undirected, g_direct):
        E = set(g_undirected.edges())
        new_edges = [e for e in g_direct.edges() if e in E or reversed(e) in E]
        return nx.DiGraph(new_edges)

    def perceptron(self, n):
        w = np.zeros(self.feature_num, dtype=int)
        for i in range(0, n):
            iteration_time = datetime.now()
            for data in self.scored_graphs:
                weighted_full_graph = self.get_weighted_graph(data[2], data[3], w)
                g_tag_undirected = nx.algorithms.minimum_spanning_tree(weighted_full_graph.to_undirected())
                g_tag = self.get_directed_graph(g_tag_undirected, weighted_full_graph)

                if True:  # TODO: NEED HERE TO CHECK IF NOT EQUAL?
                    w = w + self.get_f_vector_real(data[0]) - self.get_f_vector(g_tag)
            print('Done ' + str(i+1) + ' iteration at ' + str(datetime.now()-iteration_time))
        return w

    #########
    # Train #
    #########

    def train(self):
        start_time = datetime.now()
        print('\nGetting all features...')
        self.get_features()
        self.build_graphs()
        print('Found the following features:')
        print('------------------------------')
        print(str(len(self.feature_1_dict)) + ' of feature 1')
        print(str(len(self.feature_2_dict)) + ' of feature 2')
        print(str(len(self.feature_3_dict)) + ' of feature 3')
        print(str(len(self.feature_4_dict)) + ' of feature 4')
        print(str(len(self.feature_5_dict)) + ' of feature 5')
        print(str(len(self.feature_6_dict)) + ' of feature 6')
        print(str(len(self.feature_8_dict)) + ' of feature 8')
        print(str(len(self.feature_10_dict)) + ' of feature 10')
        print(str(len(self.feature_13_dict)) + ' of feature 13')
        print('***Total of ' + str(len(self.feature_1_dict)+len(self.feature_2_dict)+len(self.feature_3_dict) +
                                   len(self.feature_4_dict)+len(self.feature_5_dict)+len(self.feature_6_dict) +
                                   len(self.feature_8_dict)+len(self.feature_10_dict)+len(self.feature_13_dict)) +
              ' features***')

        print('\nAfter optimization, we left with the following features:')
        print('----------------------------------------------------------')
        self.get_frequent_features()
        print('***Total of ' + str(len(self.features)) + ' optimized features***')

        print('\nDoing some magic...')
        self.calculate_features_for_all_graphs()
        print('All done!')

        print('\nStarting perceptron with N = 20:')
        w_20 = self.perceptron(20)
        print('DONE perceptron (N=20)')
        pickle.dump(w_20, open("Perceptron Results\\basic_w_20.p", "wb"), protocol=2)

        print('\nStarting perceptron with N = 50:')
        w_50 = self.perceptron(50)
        print('DONE perceptron (N=50)')
        pickle.dump(w_50, open("Perceptron Results\\basic_w_50.p", "wb"), protocol=2)

        print('\nStarting perceptron with N = 80:')
        w_80 = self.perceptron(80)
        print('DONE perceptron (N=80)')
        pickle.dump(w_80, open("Perceptron Results\\basic_w_80.p", "wb"), protocol=2)

        print('\nStarting perceptron with N = 100:')
        w_100 = self.perceptron(100)
        print('DONE perceptron (N=100)')
        pickle.dump(w_100, open("Perceptron Results\\basic_w_100.p", "wb"), protocol=2)

        print('\nTHE LEARNING PROCESS TOOK ' + str(datetime.now()-start_time))

x = DependencyTrainer()
x.train()
