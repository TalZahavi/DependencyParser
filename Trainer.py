from datetime import datetime
import pickle
import numpy as np
import edmonds
import random


class Trainer:
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

        self.arches_data_list = []

        self.features = dict()
        self.feature_num = 0

        self.graphs = []  # Holds (graph, full_graph) #TODO: FIX COMMENT
        self.scored_graphs = []  # Holds (graph, graph score,  full_graph, full_graph score) #TODO: FIX COMMENT

        self.saved_f_vector = dict()

    #################
    # FEATURES PART #
    #################

    # Add all seen features to the dicts
    def get_features(self):
        sentence_words_pos = dict()
        with open('Data\\train.labeled', 'r') as f:
            for line in f:
                if line is '\n':
                    arches = []
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

                        arches.append((dependency_arch, word_tuple[2], counter))
                        # data_for_full_graph.add((dependency_arch[0], dependency_arch[1]))
                        data_for_full_graph.add((dependency_arch[2], dependency_arch[3], counter))

                    arches.append(('root', 'root', 0))
                    data_for_full_graph.add(('root', 'root', 0))
                    self.arches_data_list.append((arches, list(data_for_full_graph)))
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

    ##############
    # GRAPH PART #
    ##############

    def make_graphs(self):
        for arches_data in self.arches_data_list:
            # Get (arches tuples, full data)
            g = dict()
            arches_tuples = arches_data[0]

            for arch_tuple in arches_tuples:
                if (arch_tuple[0][0], arch_tuple[0][1], arch_tuple[1]) in g:
                    g[(arch_tuple[0][0], arch_tuple[0][1], arch_tuple[1])][(arch_tuple[0][2], arch_tuple[0][3], arch_tuple[2])] = 0
                else:
                    g[(arch_tuple[0][0], arch_tuple[0][1], arch_tuple[1])] = {(arch_tuple[0][2], arch_tuple[0][3], arch_tuple[2]): 0}

            self.graphs.append((g, self.make_full_graph(arches_data[1])))

    @staticmethod
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
    # TODO: CHECK AGAIN!
    def get_features_for_graph(self, g):
        features_list = []
        for head_data in g:
            for child_data in g[head_data]:
                dependency_arch = (head_data[0], head_data[1], child_data[0], child_data[1])
                features_list.append(((head_data[0], head_data[1], head_data[2]), (child_data[0], child_data[1], child_data[2]),
                                      self.get_features_for_arch(dependency_arch)))
        return features_list

    # Doing this calculation one time !
    # checking the features for each graph (real and full)
    def calculate_features_for_all_graphs(self):
        graph_number = 0  # We use this to later save f_vector result in dict
        for data in self.graphs:
            graph_number += 1
            self.scored_graphs.append((data[0], self.get_features_for_graph(data[0]),
                                       data[1], self.get_features_for_graph(data[1]), graph_number))

    ########################
    # Perceptron Algorithm #
    ########################

    @staticmethod
    def get_weighted_graph(g, features_list, w):
        for features_edge in features_list:
            weight = 0
            for feature_i in features_edge[2]:
                weight += w[feature_i]
            g[(features_edge[0][0], features_edge[0][1], features_edge[0][2])][(features_edge[1][0], features_edge[1][1], features_edge[1][2])] = -weight
        return g

    def get_f_vector(self, g):
        f_vector = np.zeros(self.feature_num, dtype=int)
        features = self.get_features_for_graph(g)
        for feature_data in features:
            for feature_i in feature_data[2]:
                f_vector[feature_i] += 1
        return f_vector

    def get_saved_f_vector(self, g, g_num):
        if g_num in self.saved_f_vector:
            return self.saved_f_vector[g_num]
        else:
            temp = self.get_f_vector(g)
            self.saved_f_vector[g_num] = temp
            return temp

    def perceptron(self, n):
        w = np.zeros(self.feature_num, dtype=int)
        for i in range(0, n):
            iteration_time = datetime.now()
            scored_graph_index = list(range(0, len(self.scored_graphs), 1))
            shuffled_scored_graph_index = sorted(scored_graph_index, key=lambda k: random.random())
            for index in shuffled_scored_graph_index:
                data = self.scored_graphs[index]
                weighted_full_graph = self.get_weighted_graph(data[2], data[3], w)
                g_tag = edmonds.mst(('root', 'root', 0), weighted_full_graph)
                if True:  # TODO: NEED HERE TO CHECK IF NOT EQUAL?  + GET OUT AUX
                    w = w + self.get_saved_f_vector(data[0], data[4]) - self.get_f_vector(g_tag)
            print('Done ' + str(i+1) + ' iteration at ' + str(datetime.now()-iteration_time))
            self.save_w(w, i+1)

    @staticmethod
    def save_w(w, iteration):
        if iteration == 20:
            print('DONE perceptron (N=20)\n')
            pickle.dump(w, open("Perceptron Results\\basic_w_20.p", "wb"), protocol=2)
        if iteration == 50:
            print('DONE perceptron (N=50)\n')
            pickle.dump(w, open("Perceptron Results\\basic_w_50.p", "wb"), protocol=2)
        if iteration == 80:
            print('DONE perceptron (N=80)\n')
            pickle.dump(w, open("Perceptron Results\\basic_w_80.p", "wb"), protocol=2)
        if iteration == 100:
            print('DONE perceptron (N=100)\n')
            pickle.dump(w, open("Perceptron Results\\basic_w_100.p", "wb"), protocol=2)

    #########
    # Train #
    #########

    def train(self):
        start_time = datetime.now()
        print('\nGetting all features...')
        self.get_features()
        self.make_graphs()
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

        pickle.dump(self.features, open("Perceptron Results\\features.p", "wb"), protocol=2)

        print('\nDoing some magic...')
        self.calculate_features_for_all_graphs()
        print('All done!')

        print('\nStarting perceptron...')
        self.perceptron(100)
        print('DONE')

        print('\nTHE LEARNING PROCESS TOOK ' + str(datetime.now()-start_time))


x = Trainer()
x.train()
