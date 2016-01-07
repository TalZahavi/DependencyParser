import pickle


# Get a dependency arch, and add all the possible features to the dict
def add_features_to_dicts(self, dependency_arch):

    # Basic model function
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

    # Another basic functions
    self.add_feature_for_dependency(((dependency_arch[0], dependency_arch[1]), (dependency_arch[2], dependency_arch[3])), self.feature_7_dict)
    self.add_feature_for_dependency((dependency_arch[0], (dependency_arch[2], dependency_arch[3])), self.feature_9_dict)
    self.add_feature_for_dependency((dependency_arch[0], (dependency_arch[1], dependency_arch[2])), self.feature_11_dict)
    self.add_feature_for_dependency((dependency_arch[0], dependency_arch[2]), self.feature_12_dict)


# Get a counter to tags and add all the improved features to the dict
def add_improved_features_to_dicts(self, sentence_words_pos):
    for c_counter in sentence_words_pos:
        c_tuple = sentence_words_pos[c_counter]
        p_counter = c_tuple[2]

        # p-pos,p-pos+1,c-pos-1,c-pos
        if c_counter != 0 and p_counter != len(sentence_words_pos) - 1:
            self.add_feature_for_dependency(((sentence_words_pos[p_counter][1], sentence_words_pos[p_counter + 1][1]),
                                            (sentence_words_pos[c_counter - 1][1], sentence_words_pos[c_counter][1])),
                                            self.feature_14_dict)

        # p-pos-1,p-pos,c-pos-1,c-pos
        if c_counter != 0 and p_counter != 0:
            self.add_feature_for_dependency(((sentence_words_pos[p_counter-1][1], sentence_words_pos[p_counter][1]),
                                            (sentence_words_pos[c_counter-1][1], sentence_words_pos[c_counter][1])),
                                            self.feature_15_dict)

        # p-pos,p-pos+1,c-pos,c-pos+1
        if c_counter != len(sentence_words_pos) - 1 and p_counter != len(sentence_words_pos) - 1:
            self.add_feature_for_dependency(((sentence_words_pos[p_counter][1], sentence_words_pos[p_counter+1][1]),
                                            (sentence_words_pos[c_counter][1], sentence_words_pos[c_counter+1][1])),
                                             self.feature_16_dict)

        # p-pos-1,p-pos,c-pos,c-pos+1
        if c_counter != len(sentence_words_pos) - 1 and p_counter != 0:
            self.add_feature_for_dependency(((sentence_words_pos[p_counter - 1][1], sentence_words_pos[p_counter][1]),
                                            (sentence_words_pos[c_counter][1], sentence_words_pos[c_counter+1][1])),
                                             self.feature_17_dict)
        # p-pos,b-pos,c-pos
        if c_counter < p_counter:
            for index in range(c_counter+1, p_counter):
                self.add_feature_for_dependency((sentence_words_pos[p_counter][1], (sentence_words_pos[index][1],
                                            sentence_words_pos[c_counter][1])), self.feature_18_dict)
        else:
            for index in range(p_counter+1, c_counter):
                self.add_feature_for_dependency((sentence_words_pos[p_counter][1], (sentence_words_pos[index][1],
                                            sentence_words_pos[c_counter][1])), self.feature_18_dict)


# Return the features number that return 1 for a given arch
def get_features_for_arch(features, dependency_arch, g_words_tags, c_counter, p_counter):
    num_features = []
    if p_counter == 'root':
        p_counter = 0
    # Basic functions
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

    # Improved basic functions
    if (((dependency_arch[0], dependency_arch[1]), (dependency_arch[2], dependency_arch[3])), 7) in features:
        num_features.append(features[(((dependency_arch[0], dependency_arch[1]), (dependency_arch[2], dependency_arch[3])), 7)])
    if ((dependency_arch[0], (dependency_arch[2], dependency_arch[3])), 9) in features:
        num_features.append(features[((dependency_arch[0], (dependency_arch[2], dependency_arch[3])), 9)])
    if ((dependency_arch[0], (dependency_arch[1], dependency_arch[2])), 11) in features:
        num_features.append(features[((dependency_arch[0], (dependency_arch[1], dependency_arch[2])), 11)])
    if ((dependency_arch[0], dependency_arch[2]), 12) in features:
        num_features.append(features[((dependency_arch[0], dependency_arch[2]), 12)])

    # Improved functions
    # p-pos,p-pos+1,c-pos-1,c-pos
    if c_counter != 0 and p_counter != len(g_words_tags) - 1:
        if (((g_words_tags[p_counter], g_words_tags[p_counter+1]), (g_words_tags[c_counter - 1],
                        g_words_tags[c_counter])), 14) in features:
            num_features.append(features[(((g_words_tags[p_counter], g_words_tags[p_counter+1]), (g_words_tags[c_counter - 1],
                        g_words_tags[c_counter])), 14)])

    # p-pos-1,p-pos,c-pos-1,c-pos
    if c_counter != 0 and p_counter != 0:
        if (((g_words_tags[p_counter-1], g_words_tags[p_counter]), (g_words_tags[c_counter - 1],
                        g_words_tags[c_counter])), 15) in features:
            num_features.append(features[(((g_words_tags[p_counter-1], g_words_tags[p_counter]), (g_words_tags[c_counter - 1],
                        g_words_tags[c_counter])), 15)])

    # p-pos,p-pos+1,c-pos,c-pos+1
    if c_counter != len(g_words_tags) - 1 and p_counter != len(g_words_tags) - 1:
        if (((g_words_tags[p_counter], g_words_tags[p_counter+1]), (g_words_tags[c_counter],
                        g_words_tags[c_counter+1])), 16) in features:
            num_features.append(features[(((g_words_tags[p_counter], g_words_tags[p_counter+1]), (g_words_tags[c_counter],
                        g_words_tags[c_counter+1])), 16)])

    # p-pos-1,p-pos,c-pos,c-pos+1
    if c_counter != len(g_words_tags) - 1 and p_counter != 0:
        if (((g_words_tags[p_counter-1], g_words_tags[p_counter]), (g_words_tags[c_counter],
                        g_words_tags[c_counter+1])), 17) in features:
            num_features.append(features[(((g_words_tags[p_counter-1], g_words_tags[p_counter]), (g_words_tags[c_counter],
                        g_words_tags[c_counter+1])), 17)])

    # p-pos,b-pos,c-pos
    if c_counter < p_counter:
        for index in range(c_counter+1, p_counter):
            if ((g_words_tags[p_counter], (g_words_tags[index],
                        g_words_tags[c_counter])), 18) in features:
                num_features.append(features[((g_words_tags[p_counter], (g_words_tags[index],
                        g_words_tags[c_counter])), 18)])
    else:
        for index in range(p_counter+1, c_counter):
            if ((g_words_tags[p_counter], (g_words_tags[index],
                        g_words_tags[c_counter])), 18) in features:
                num_features.append(features[((g_words_tags[p_counter], (g_words_tags[index],
                        g_words_tags[c_counter])), 18)])

    return num_features


# Get only frequent features (configurable limits)
def get_frequent_features(self):
    # Basic function
    self.add_frequent_feature(self.FEATURE_1_LIMIT, self.feature_1_dict, 1)
    self.add_frequent_feature(self.FEATURE_2_LIMIT, self.feature_2_dict, 2)
    self.add_frequent_feature(self.FEATURE_3_LIMIT, self.feature_3_dict, 3)
    self.add_frequent_feature(self.FEATURE_4_LIMIT, self.feature_4_dict, 4)
    self.add_frequent_feature(self.FEATURE_5_LIMIT, self.feature_5_dict, 5)
    self.add_frequent_feature(self.FEATURE_6_LIMIT, self.feature_6_dict, 6)
    self.add_frequent_feature(self.FEATURE_8_LIMIT, self.feature_8_dict, 8)
    self.add_frequent_feature(self.FEATURE_10_LIMIT, self.feature_10_dict, 10)
    self.add_frequent_feature(self.FEATURE_13_LIMIT, self.feature_13_dict, 13)

    # Improved basic function
    self.add_frequent_feature(self.FEATURE_7_LIMIT, self.feature_7_dict, 7)
    self.add_frequent_feature(self.FEATURE_9_LIMIT, self.feature_9_dict, 9)
    self.add_frequent_feature(self.FEATURE_11_LIMIT, self.feature_11_dict, 11)
    self.add_frequent_feature(self.FEATURE_12_LIMIT, self.feature_12_dict, 12)

    # Improved  function
    self.add_frequent_feature(self.FEATURE_14_LIMIT, self.feature_14_dict, 14)
    self.add_frequent_feature(self.FEATURE_15_LIMIT, self.feature_15_dict, 15)
    self.add_frequent_feature(self.FEATURE_16_LIMIT, self.feature_16_dict, 16)
    self.add_frequent_feature(self.FEATURE_17_LIMIT, self.feature_17_dict, 17)
    self.add_frequent_feature(self.FEATURE_18_LIMIT, self.feature_18_dict, 18)


# If its the 20\50\80\100 iteration - save the w vector for later use
def save_w(w, iteration):
    if iteration == 20:
        print('DONE perceptron (N=20)\n')
        pickle.dump(w, open("Perceptron Results\\Improved_w_20.p", "wb"), protocol=2)
    if iteration == 50:
        print('DONE perceptron (N=50)\n')
        pickle.dump(w, open("Perceptron Results\\Improved_w_50.p", "wb"), protocol=2)
    if iteration == 80:
        print('DONE perceptron (N=80)\n')
        pickle.dump(w, open("Perceptron Results\\Improved_w_80.p", "wb"), protocol=2)
    if iteration == 100:
        print('DONE perceptron (N=100)\n')
        pickle.dump(w, open("Perceptron Results\\Improved_w_100.p", "wb"), protocol=2)
