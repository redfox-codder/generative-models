from sklearn.mixture import BayesianGaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

from tensorflow.examples.tutorials.mnist import input_data

BayesianGaussianMixture().sample

class GMM_Classifier:
    def fit(self, X, Y):
        self.K = len(set (Y))

        self.GMM = []

        for k in range(self.K):
            Xk = X[Y==k]

            gmm = BayesianGaussianMixture(10)
            gmm.fit(Xk)

            self.GMM.append(gmm)

    def sample_given_y(self, y):
        gmm =self.GMM[y]
        sample = gmm.sample()
        # note: sample returns a tuple containing 2 things:
        # 1) the sample
        # 2) which cluster it came from
        # we'll use (2) to obtain the means so we can plot
        # them like we did in the previous script
        # we cheat by looking at "non-public" params in
        # the sklearn source code

        #%% This command is strange
        mean = gmm.means_[sample[1]]


        return sample[0].reshape(28, 28), mean.reshape(28, 28)

    def sample(self):
        y = np.random.choice(self.K)
        return self.sample(y)


if __name__ == '__main__':
    # X, Y = util.get_mnist()
    mnist = input_data.read_data_sets(train_dir="../03-Convolutional-Neural-Networks/MNIST_data/", one_hot=False)

    X = mnist.train.images
    Y = mnist.train.labels

    clf = GMM_Classifier()
    clf.fit(X, Y)

    for k in range(clf.K):
        # show one sample for each class
        # also show the mean image learned

        sample, mean = clf.sample_given_y(k)

        plt.subplot(1, 2, 1)
        plt.imshow(sample, cmap='gray')
        plt.title("Sample")
        plt.subplot(1, 2, 2)
        plt.imshow(mean, cmap='gray')
        plt.title("Mean")
        plt.show()

    # generate a random sample
    sample, mean = clf.sample()
    plt.subplot(1, 2, 1)
    plt.imshow(sample, cmap='gray')
    plt.title("Random Sample from Random Class")
    plt.subplot(1, 2, 2)
    plt.imshow(mean, cmap='gray')
    plt.title("Corresponding Cluster Mean")
    plt.show()


