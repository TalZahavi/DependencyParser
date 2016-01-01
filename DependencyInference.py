import pickle


class DependencyInference:
    def __init__(self):
        self.w_20 = self.tags = pickle.load(open("Perceptron Results\\basic_w_20.p", "rb"))
        self.w_50 = self.tags = pickle.load(open("Perceptron Results\\basic_w_50.p", "rb"))
        self.w_80 = self.tags = pickle.load(open("Perceptron Results\\basic_w_80.p", "rb"))
        self.w_100 = self.tags = pickle.load(open("Perceptron Results\\basic_w_100.p", "rb"))

    # Get labeled document, and return unlabeled document (for checking test accuracy)
    def make_unlabeled_data(self, document_name):
        pass

    # This is the inference process
    # Get unlabeled document, inference the dependencies and write to a new document
    def data_inference(self, unlabeled_document_name, output_name):
        pass

    # Get two labeled documents and return the accuracy of the test document
    def get_accuracy(self, gold_document, test_document):
        pass
