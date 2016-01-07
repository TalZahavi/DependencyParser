from datetime import datetime
import pickle
import numpy as np
import edmonds
import random
import AuxFunctions
import BasicFunctions
import ImprovedFunctions


class Trainer:

    # Basic model
    FEATURE_1_LIMIT = 0   # p - word, p-pos
    FEATURE_2_LIMIT = 0   # p - word
    FEATURE_3_LIMIT = 0   # p - pos
    FEATURE_4_LIMIT = 0   # c - word, c-pos
    FEATURE_5_LIMIT = 0   # c - word
    FEATURE_6_LIMIT = 0   # c - pos
    FEATURE_8_LIMIT = 0   # p - pos, c - word, c - pos
    FEATURE_10_LIMIT = 0  # p -word, p - pos, c - pos
    FEATURE_13_LIMIT = 0  # p - pos, c - pos

    # Improved model

    FEATURE_7_LIMIT = 0   # p -word, p - pos, c- word, c - pos
    FEATURE_9_LIMIT = 1   # p - word, c - word, c - pos
    FEATURE_11_LIMIT = 1  # p -word, p - pos, c - word
    FEATURE_12_LIMIT = 1  # p - word, c - word

    FEATURE_14_LIMIT = 0  # p-pos,p-pos+1,c-pos-1,c-pos
    FEATURE_15_LIMIT = 0  # p-pos-1,p-pos,c-pos-1,c-pos
    FEATURE_16_LIMIT = 0  # p-pos,p-pos+1,c-pos,c-pos+1
    FEATURE_17_LIMIT = 0  # p-pos-1,p-pos,c-pos,c-pos+1
    FEATURE_18_LIMIT = 0  # p-pos,b-pos,c-pos

    def __init__(self, is_improved):
        self.is_improved = is_improved
        self.feature_1_dict = dict()
        self.feature_2_dict = dict()
        self.feature_3_dict = dict()
        self.feature_4_dict = dict()
        self.feature_5_dict = dict()
        self.feature_6_dict = dict()
        self.feature_8_dict = dict()
        self.feature_10_dict = dict()
        self.feature_13_dict = dict()

        # Add basic features for improved model
        self.feature_7_dict = dict()
        self.feature_9_dict = dict()
        self.feature_11_dict = dict()
        self.feature_12_dict = dict()

        # Add improved features for improved model
        self.feature_14_dict = dict()
        self.feature_15_dict = dict()
        self.feature_16_dict = dict()
        self.feature_17_dict = dict()
        self.feature_18_dict = dict()

        self.arches_data_list = []
        self.sentence_tags = []

        self.features = dict()
        self.feature_num = 0

        self.graphs = []  # Holds (graph, full_graph)
        self.scored_graphs = []  # Holds (graph, graph features,  full_graph, full_graph features)

        self.saved_f_vector = dict()

    #################
    # FEATURES PART #
    #################

    # Add all seen features to the dicts and save useful data for the building graphs process
    def get_features(self):
        sentence_words_pos = dict()
        with open('Data\\train.labeled', 'r') as f:
            for line in f:
                if line == '\n':
                    arches = []
                    words_pos = dict()
                    data_for_full_graph = set()
                    for counter in sentence_words_pos:
                        word_tuple = sentence_words_pos[counter]
                        # dependency_arch= (head, head_pos, child, child_pos)
                        if word_tuple[2] == 0:
                            dependency_arch = ('root', 'root', word_tuple[0], word_tuple[1])
                        else:
                            dependency_arch = (sentence_words_pos[word_tuple[2]][0],
                                               sentence_words_pos[word_tuple[2]][1], word_tuple[0], word_tuple[1])
                        if self.is_improved:
                            ImprovedFunctions.add_features_to_dicts(self, dependency_arch)
                        else:
                            BasicFunctions.add_features_to_dicts(self, dependency_arch)

                        arches.append((dependency_arch, word_tuple[2], counter))
                        data_for_full_graph.add((dependency_arch[2], dependency_arch[3], counter))
                        words_pos[counter] = word_tuple[1]
                    words_pos[0] = 'root'
                    sentence_words_pos[0] = ('root', 'root', 0)
                    if self.is_improved:
                        ImprovedFunctions.add_improved_features_to_dicts(self,sentence_words_pos)
                    arches.append(('root', 'root', 0))
                    data_for_full_graph.add(('root', 'root', 0))
                    self.arches_data_list.append((arches, list(data_for_full_graph)))
                    self.sentence_tags.append(words_pos)
                    sentence_words_pos = dict()
                else:
                    split_line = line.split('\t')
                    # (counter)->(token,pos,head)
                    sentence_words_pos[int(split_line[0])] = (split_line[1], split_line[3], int(split_line[6]))
        f.close()

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

    # Make a list of (correct graph, full graph) for learning algorithm
    def make_graphs(self):
        counter = 0
        for arches_data in self.arches_data_list:
            # Get (arches tuples, full data)
            g = dict()
            arches_tuples = arches_data[0]

            for arch_tuple in arches_tuples:
                if (arch_tuple[0][0], arch_tuple[0][1], arch_tuple[1]) in g:
                    g[(arch_tuple[0][0], arch_tuple[0][1], arch_tuple[1])][(arch_tuple[0][2], arch_tuple[0][3], arch_tuple[2])] = 0
                else:
                    g[(arch_tuple[0][0], arch_tuple[0][1], arch_tuple[1])] = {(arch_tuple[0][2], arch_tuple[0][3], arch_tuple[2]): 0}

            self.graphs.append((g, AuxFunctions.make_full_graph(arches_data[1])))

    ##########################
    # FREQUENT FEATURES PART #
    ##########################

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

    # Doing this calculation one time !
    # checking the features for each graph (real and full)
    def calculate_features_for_all_graphs(self):
        graph_number = 0  # We use this to later save f_vector result in dict
        for data in self.graphs:
            graph_number += 1
            self.scored_graphs.append((data[0], AuxFunctions.get_features_for_graph(self.features, data[0],
                                    self.sentence_tags[graph_number - 1], self.is_improved), data[1],
                                    AuxFunctions.get_features_for_graph(self.features, data[1],
                                    self.sentence_tags[graph_number - 1], self.is_improved), graph_number))

    ########################
    # Perceptron Algorithm #
    ########################

    # Return the f vector for a given graph
    def get_f_vector(self, g, g_num):
        f_vector = np.zeros(self.feature_num, dtype=int)
        features = AuxFunctions.get_features_for_graph(self.features, g, self.sentence_tags[g_num-1], self.is_improved)
        for feature_data in features:
            for feature_i in feature_data[2]:
                f_vector[feature_i] += 1
        return f_vector

    # Check if we already calculated the f vector of real graph
    # If yes - return the calculated vector. if no - calculate and save the result
    def get_saved_f_vector(self, g, g_num):
        if g_num in self.saved_f_vector:
            return self.saved_f_vector[g_num]
        else:
            temp = self.get_f_vector(g, g_num)
            self.saved_f_vector[g_num] = temp
            return temp

    # This is the heart of the algorithm!
    # We are using permutation each iteration for better performance
    def perceptron(self, n):
        w = np.zeros(self.feature_num, dtype=int)
        for i in range(0, n):
            iteration_time = datetime.now()
            scored_graph_index = list(range(0, len(self.scored_graphs), 1))
            shuffled_scored_graph_index = sorted(scored_graph_index, key=lambda k: random.random())
            for index in shuffled_scored_graph_index:
                data = self.scored_graphs[index]
                weighted_full_graph = AuxFunctions.get_weighted_graph(data[2], data[3], w)
                g_tag = edmonds.mst(('root', 'root', 0), weighted_full_graph)
                if True:  # For better performance
                    w = w + self.get_saved_f_vector(data[0], data[4]) - self.get_f_vector(g_tag, data[4])
            print('Done ' + str(i+1) + ' iteration at ' + str(datetime.now()-iteration_time))
            if self.is_improved:
                ImprovedFunctions.save_w(w, i+1)
            else:
                BasicFunctions.save_w(w, i+1)

    #########
    # Train #
    #########

    # Just a method to run the training process from top to bottom
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

        num_features = len(self.feature_1_dict)+len(self.feature_2_dict)+len(self.feature_3_dict) + \
                                   len(self.feature_4_dict)+len(self.feature_5_dict)+len(self.feature_6_dict) + \
                                   len(self.feature_8_dict)+len(self.feature_10_dict)+len(self.feature_13_dict)
        if self.is_improved:
            print(str(len(self.feature_7_dict)) + ' of feature 7')
            print(str(len(self.feature_9_dict)) + ' of feature 9')
            print(str(len(self.feature_11_dict)) + ' of feature 11')
            print(str(len(self.feature_12_dict)) + ' of feature 12')
            print(str(len(self.feature_14_dict)) + ' of feature 14')
            print(str(len(self.feature_15_dict)) + ' of feature 15')
            print(str(len(self.feature_16_dict)) + ' of feature 16')
            print(str(len(self.feature_17_dict)) + ' of feature 17')
            print(str(len(self.feature_18_dict)) + ' of feature 18')
            num_features = num_features + len(self.feature_7_dict)+len(self.feature_9_dict)+len(self.feature_11_dict) \
                           + len(self.feature_12_dict) + len(self.feature_14_dict)+len(self.feature_15_dict)\
                           +len(self.feature_16_dict) + len(self.feature_17_dict) + len(self.feature_18_dict)

        print('***Total of ' + str(num_features) + ' features***')

        print('\nAfter optimization, we left with the following features:')
        print('----------------------------------------------------------')
        if self.is_improved:
            ImprovedFunctions.get_frequent_features(self)
        else:
            BasicFunctions.get_frequent_features(self)
        print('***Total of ' + str(len(self.features)) + ' optimized features***')

        if self.is_improved:
            pickle.dump(self.features, open("Perceptron Results\\features.p", "wb"), protocol=2)
        else:
            pickle.dump(self.features, open("Perceptron Results\\basic_features.p", "wb"), protocol=2)

        print('\nDoing some magic...')
        self.calculate_features_for_all_graphs()
        print('All done!')

        print('\nStarting perceptron...')
        self.perceptron(100)
        print('DONE')

        print('\nTHE LEARNING PROCESS TOOK ' + str(datetime.now()-start_time))


x = Trainer(True)
x.train()
