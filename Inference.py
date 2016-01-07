import pickle
import edmonds
import AuxFunctions


class Inference:

    def __init__(self, is_improved):
        self.is_improved = is_improved
        if self.is_improved:
            self.features = pickle.load(open("Perceptron Results\\features.p", "rb"))
            self.w_20 = pickle.load(open("Perceptron Results\\Improved_w_20.p", "rb"))
            self.w_50 = pickle.load(open("Perceptron Results\\Improved_w_50.p", "rb"))
            self.w_80 = pickle.load(open("Perceptron Results\\Improved_w_80.p", "rb"))
            self.w_100 = pickle.load(open("Perceptron Results\\Improved_w_100.p", "rb"))
        else:
            self.w_20 = pickle.load(open("Perceptron Results\\basic_w_20.p", "rb"))
            self.w_50 = pickle.load(open("Perceptron Results\\basic_w_50.p", "rb"))
            self.w_80 = pickle.load(open("Perceptron Results\\basic_w_80.p", "rb"))
            self.w_100 = pickle.load(open("Perceptron Results\\basic_w_100.p", "rb"))
            self.features = pickle.load(open("Perceptron Results\\basic_features.p", "rb"))

    # Get a graph and return arches in the right order (by token number)
    @staticmethod
    def get_arches_in_order(g):
        ordered_arches = dict()
        for node_head in g:
            for node_child in g[node_head]:
                word_num = node_child[2]
                ordered_arches[word_num] = (node_child[0], node_child[1], node_head[2])
        return ordered_arches

    # Get ordered data of the graph, and write the lines for the accuracy check
    @staticmethod
    def write_lines(lines_dict, f):
        for i in range(1, len(lines_dict)+1):
                f.write(str(i) + '\t' + str(lines_dict[i][0]) + '\t_\t' + str(lines_dict[i][1]) + '\t_\t_\t' +
                        str(lines_dict[i][2]) + '\t_\t_\t_\n')

        f.write('\n')

    # Get unlabeled document, inference the dependencies and write to a new document
    def data_inference(self, unlabeled_document_name, output_name, w):
        sentence_words_pos = dict()
        sentence_num = 0
        with open(unlabeled_document_name, 'r') as f1:
            with open(output_name, 'w+') as f2:
                for line in f1:

                    if line == '\n':
                        data_for_full_graph = set()
                        words_tags = dict()
                        for counter in sentence_words_pos:
                            word_tuple = sentence_words_pos[counter]
                            data_for_full_graph.add((word_tuple[0], word_tuple[1], counter))
                            words_tags[counter] = sentence_words_pos[counter][1]

                        data_for_full_graph.add(('root', 'root', 0))
                        sentence_words_pos[0] = ('root', 'root', 0)
                        words_tags[0] = 'root'

                        full_unweighted_g = AuxFunctions.make_full_graph(list(data_for_full_graph))
                        g_features = AuxFunctions.get_features_for_graph(self.features, full_unweighted_g, words_tags, self.is_improved)
                        full_weighted_g = AuxFunctions.get_weighted_graph(full_unweighted_g, g_features, w)

                        g_inference = edmonds.mst(('root', 'root', 0), full_weighted_g)

                        inference_arches = self.get_arches_in_order(g_inference)
                        self.write_lines(inference_arches, f2)

                        sentence_num += 1
                        print('Done sentence number ' + str(sentence_num))
                        sentence_words_pos = dict()
                    else:
                        split_line = line.split('\t')
                        # (counter)->(token,pos)
                        sentence_words_pos[int(split_line[0])] = (split_line[1], split_line[3])

            f2.close()
        f1.close()

    # Get two labeled documents and return the accuracy of the test document
    @staticmethod
    def get_accuracy(gold_document, test_document):
        total = 0
        correct = 0
        sentence_num = 0
        with open(gold_document, 'r') as f1, open(test_document, 'r') as f2:
            for line1, line2 in zip(f1, f2):
                if line2 == '\n':
                    sentence_num += 1
                    # print('The accuracy after the ' + str(sentence_num) + ' sentence is: ' +
                    #       str(round((correct/total)*100, 2)) + '%')
                else:
                    gold_split = line1.split('\t')
                    test_split = line2.split('\t')
                    total += 1
                    if gold_split[0] == test_split[0] and gold_split[1] == test_split[1] and\
                            gold_split[3] == test_split[3]:
                                if gold_split[6] == test_split[6]:
                                    correct += 1
        f1.close()
        f2.close()
        return float(correct)/float(total)*100

z = Inference(True)
print('Starting inference for w_20...')
z.data_inference('Data\\test.unlabeled', 'Data\\test.mylabel20', z.w_20)
result_20 = z.get_accuracy('Data\\test.labeled', 'Data\\test.mylabel20')

print('\nStarting inference for w_50...')
z.data_inference('Data\\test.unlabeled', 'Data\\test.mylabel50', z.w_50)
result_50 = z.get_accuracy('Data\\test.labeled', 'Data\\test.mylabel50')

print('\nStarting inference for w_80...')
z.data_inference('Data\\test.unlabeled', 'Data\\test.mylabel80', z.w_80)
result_80 = z.get_accuracy('Data\\test.labeled', 'Data\\test.mylabel80')

print('\nStarting inference for w_100...')
z.data_inference('Data\\test.unlabeled', 'Data\\test.mylabel100', z.w_100)
result_100 = z.get_accuracy('Data\\test.labeled', 'Data\\test.mylabel100')

print('\nThe accuracy for w_20 is ' + str(result_20) + '%')
print('The accuracy for w_50 is ' + str(result_50) + '%')
print('The accuracy for w_80 is ' + str(result_80) + '%')
print('The accuracy for w_100 is ' + str(result_100) + '%')
