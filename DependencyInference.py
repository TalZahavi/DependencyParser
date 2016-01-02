import pickle


class DependencyInference:
    def __init__(self):
        self.w_20 = self.tags = pickle.load(open("Perceptron Results\\basic_w_20.p", "rb"))
        self.w_50 = self.tags = pickle.load(open("Perceptron Results\\basic_w_50.p", "rb"))
        self.w_80 = self.tags = pickle.load(open("Perceptron Results\\basic_w_80.p", "rb"))
        self.w_100 = self.tags = pickle.load(open("Perceptron Results\\basic_w_100.p", "rb"))

    # Get labeled document, and return unlabeled document (for checking test accuracy)
    @staticmethod
    def make_unlabeled_data(document_name):
        f_w = open('Data\\test.unlabeled', 'w+')
        with open(document_name, 'r') as f:
            for line in f:
                if line is '\n':
                    f_w.write(line)
                else:
                    line_split = line.split('\t')
                    f_w.write(line_split[0] + '\t' + line_split[1] + '\t-\t' + line_split[3] + '\t-\t-\t-\t-\t-\t-\n')

        f.close()
        f_w.close()

    # This is the inference process
    # Get unlabeled document, inference the dependencies and write to a new document
    def data_inference(self, unlabeled_document_name, output_name):
        pass

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

y = DependencyInference()

# y.make_unlabeled_data('Data\\test.labeled')
