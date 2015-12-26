from datetime import datetime

class DpTrainer:
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

        self.sentence_to_arch_dict = dict()

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


start_time = datetime.now()
x = DpTrainer()
x.get_features()
print('It took ' + str(datetime.now()-start_time))



