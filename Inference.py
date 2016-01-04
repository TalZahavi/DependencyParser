import pickle
import edmonds
import AuxFunctions


class Inference:

    def __init__(self):
        self.w_20 = pickle.load(open("Perceptron Results\\basic_w_20.p", "rb"))
        self.w_50 = pickle.load(open("Perceptron Results\\basic_w_50.p", "rb"))
        self.w_80 = pickle.load(open("Perceptron Results\\basic_w_80.p", "rb"))
        self.w_100 = pickle.load(open("Perceptron Results\\basic_w_100.p", "rb"))

        self.features = pickle.load(open("Perceptron Results\\features.p", "rb"))

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
                f.write(str(i) + '\t' + str(lines_dict[i][0]) + '\t-\t' + str(lines_dict[i][1]) + '\t-\t-\t' +
                        str(lines_dict[i][2]) + '\t-\t-\t-\n')

        f.write('\n')

    # Get unlabeled document, inference the dependencies and write to a new document
    def data_inference(self, unlabeled_document_name, output_name, w):
        sentence_words_pos = dict()
        # words_pos_to_word_num = dict()
        sentence_num = 0
        with open(unlabeled_document_name, 'r') as f1:
            with open(output_name, 'w+') as f2:
                for line in f1:

                    if line is '\n':
                        data_for_full_graph = set()
                        for counter in sentence_words_pos:
                            word_tuple = sentence_words_pos[counter]
                            data_for_full_graph.add((word_tuple[0], word_tuple[1], counter))

                        data_for_full_graph.add(('root', 'root', 0))

                        # TODO: 6. Replace Aux in Trainer

                        full_unweighted_g = AuxFunctions.make_full_graph(list(data_for_full_graph))
                        g_features = AuxFunctions.get_features_for_graph(self.features, full_unweighted_g)
                        full_weighted_g = AuxFunctions.get_weighted_graph(full_unweighted_g, g_features, w)

                        g_inference = edmonds.mst(('root', 'root', 0), full_weighted_g)

                        # words_pos_to_word_num[('root', 'root')] = 0
                        inference_arches = self.get_arches_in_order(g_inference)
                        self.write_lines(inference_arches, f2)

                        sentence_num += 1
                        print('Done sentence number ' + str(sentence_num))
                        sentence_words_pos = dict()
                        # words_pos_to_word_num = dict()
                    else:
                        split_line = line.split('\t')
                        # (counter)->(token,pos)
                        sentence_words_pos[int(split_line[0])] = (split_line[1], split_line[3])
                        # words_pos_to_word_num[(split_line[1], split_line[3])] = int(split_line[0])

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
                if line2 is '\n':
                    sentence_num += 1
                    print('The accuracy after the ' + str(sentence_num) + ' sentence is: ' +
                          str(round((correct/total)*100, 2)) + '%')
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

z = Inference()
z.data_inference('Data\\test.unlabeled', 'Data\\test.final', z.w_100)
z.get_accuracy('Data\\test.labeled', 'Data\\test.final')
