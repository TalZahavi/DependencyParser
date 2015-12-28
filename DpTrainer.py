from datetime import datetime


class DpTrainer:
    FEATURE_1_LIMIT = 0
    FEATURE_2_LIMIT = 0
    FEATURE_3_LIMIT = 0
    FEATURE_4_LIMIT = 0
    FEATURE_5_LIMIT = 0
    FEATURE_6_LIMIT = 0
    FEATURE_8_LIMIT = 0
    FEATURE_10_LIMIT = 0
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

        # TODO: Need this???
        # TODO: Ask Omer - There's sentences in the training that appear more than once!
        self.sentence_to_arch_dict = dict()

        self.features = dict()
        self.feature_num = 0

        self.calculated_trees_score = dict()

    #################
    # FEATURES PART #
    #################

    def get_features(self):
        sentence_words_pos = dict()
        with open('Data\\train.labeled', 'r') as f:
            for line in f:
                if line is '\n':
                    arches = set()
                    for counter in sentence_words_pos:
                        word_tuple = sentence_words_pos[counter]
                        if word_tuple[2] == 0:
                            # dependency_arch= (head, head_pos, child, child_pos)
                            dependency_arch = ('root', 'root', word_tuple[0], word_tuple[1])
                        else:
                            dependency_arch = (sentence_words_pos[word_tuple[2]][0],
                                               sentence_words_pos[word_tuple[2]][1], word_tuple[0], word_tuple[1])
                        self.add_features_to_dicts(dependency_arch)
                        arches.add(dependency_arch)
                    self.sentence_to_arch_dict[self.get_sentence(sentence_words_pos)] = arches
                    sentence_words_pos = dict()
                else:
                    split_line = line.split('\t')
                    # (counter)->(token,pos,head)
                    sentence_words_pos[int(split_line[0])] = (split_line[1], split_line[3], int(split_line[6]))
        f.close()

    # Get labeled data, and return only the sentence
    @staticmethod
    def get_sentence(data_tuple):
        sentence = ''
        for i in range(1, len(data_tuple)):
            sentence += data_tuple[i][0] + ' '
        sentence += data_tuple[len(data_tuple)][0]
        return sentence

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

    ##########################
    # FREQUENT FEATURES PART #
    ##########################

    def get_frequent_features(self):
        self.add_frequent_feature(self.FEATURE_1_LIMIT, self.feature_1_dict, 1)
        self.add_frequent_feature(self.FEATURE_2_LIMIT, self.feature_2_dict, 2)
        self.add_frequent_feature(self.FEATURE_3_LIMIT, self.feature_3_dict, 3)
        self.add_frequent_feature(self.FEATURE_4_LIMIT, self.feature_4_dict,4)
        self.add_frequent_feature(self.FEATURE_5_LIMIT, self.feature_5_dict, 5)
        self.add_frequent_feature(self.FEATURE_6_LIMIT, self.feature_6_dict, 6)
        self.add_frequent_feature(self.FEATURE_8_LIMIT, self.feature_8_dict, 8)
        self.add_frequent_feature(self.FEATURE_10_LIMIT, self.feature_10_dict, 10)
        self.add_frequent_feature(self.FEATURE_13_LIMIT, self.feature_13_dict, 13)

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

    def calculate_real_score(self):
        for sentence in self.sentence_to_arch_dict:
            num_features = []
            for dependency_arch in self.sentence_to_arch_dict[sentence]:
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

                self.calculated_trees_score[sentence] = num_features

    def train(self):
        start_time = datetime.now()
        print('\nGetting all features...')
        self.get_features()
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

        print('\nCalculating the score for all labeled sentence...')
        self.calculate_real_score()
        print('All done!')

        print('\nTHE LEARNING PROCESS TOOK ' + str(datetime.now()-start_time))


x = DpTrainer()
x.train()



