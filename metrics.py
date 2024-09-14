import copy

import numpy as np
from scipy.special import softmax


class CELoss(object):

    def compute_bin_boundaries(self, probabilities=np.array([])):

        # uniform bin spacing
        if probabilities.size == 0:
            bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]
        else:
            # size of bins
            bin_n = int(self.n_data / self.n_bins)

            bin_boundaries = np.array([])

            probabilities_sort = np.sort(probabilities)

            for i in range(0, self.n_bins):
                bin_boundaries = np.append(bin_boundaries, probabilities_sort[i * bin_n])
            bin_boundaries = np.append(bin_boundaries, 1.0)

            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]

    def get_probabilities(self, output, labels, threshold_param, logits):
        # If not probabilities apply softmax!
        if logits:
            self.probabilities = softmax(output, axis=1)
        else:
            self.probabilities = output

        self.labels = labels
        self.confidences = np.max(self.probabilities, axis=1)
        self.confidences[self.confidences < 0.5] = 1 - self.confidences[self.confidences < 0.5]

        self.predictions = copy.deepcopy(self.probabilities)
        self.predictions[self.predictions >= threshold_param] = 1
        self.predictions[self.predictions < threshold_param] = 0
        self.accuracies = np.equal(self.predictions, labels)

    def binary_matrices(self):
        idx = np.arange(self.n_data)

        # make matrices of zeros
        pred_matrix = np.zeros([self.n_data, self.n_class])
        label_matrix = np.zeros([self.n_data, self.n_class])
        pred_matrix[idx, self.predictions] = 1
        label_matrix[idx, self.labels] = 1

        self.acc_matrix = np.equal(pred_matrix, label_matrix)

    def compute_bins(self, index=None):
        self.bin_prop = np.zeros(self.n_bins)
        self.bin_acc = np.zeros(self.n_bins)
        self.bin_conf = np.zeros(self.n_bins)
        self.bin_score = np.zeros(self.n_bins)

        if index == None:
            confidences = self.confidences
            accuracies = self.accuracies
        else:
            confidences = self.probabilities[:, index]
            accuracies = self.acc_matrix[:, index]

        for i, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            # Calculated |confidence - accuracy| in each bin
            in_bin = np.greater(confidences, bin_lower.item()) * np.less_equal(confidences, bin_upper.item())
            self.bin_prop[i] = np.mean(in_bin)

            if self.bin_prop[i].item() > 0:
                self.bin_acc[i] = np.mean(accuracies[in_bin])
                self.bin_conf[i] = np.mean(confidences[in_bin])
                self.bin_score[i] = np.abs(self.bin_conf[i] - self.bin_acc[i])


class MaxProbCELoss(CELoss):
    def loss(self, output, labels, threshold_param, n_bins=15, logits=True):
        self.n_bins = n_bins
        super().compute_bin_boundaries()
        super().get_probabilities(output, labels, threshold_param, logits)
        super().compute_bins()


# http://people.cs.pitt.edu/~milos/research/AAAI_Calibration.pdf
class ECELoss(MaxProbCELoss):

    def loss(self, output, labels, threshold_param, n_bins=15, logits=True):
        super().loss(output, labels, threshold_param, n_bins, logits)
        return np.dot(self.bin_prop, self.bin_score)
