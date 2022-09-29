from matplotlib import pyplot as plt
from keras.datasets import cifar10

from classifiers.knn.knn import KNearstNeighbor

(traing_x, traing_y), (test_x, test_y) = cifar10.load_data()

# for i in range(9):
#     plt.subplot(330 + 1 + i)
#     plt.imshow(traing_x[i])

# plt.show()


def knn():
    knn_classifier = KNearstNeighbor()
    knn_classifier.train(
        x_train=traing_x,
        y_train=traing_y)
    knn_classifier.predict(test_x)


if __name__ == '__main__':
    knn()