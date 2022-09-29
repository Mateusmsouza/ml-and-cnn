import numpy as np

class KNearstNeighbor:

    def train(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_data, k=1):
        nearst_neighbors = []
        for x in x_data:
            distances_between_points = self.__euclidean_distance(
                point_a=x,
                points_b=self.x_train)
            distances_between_points
            sorted_distance = np.argsort(distances_between_points)
            nearst_neighbors.append(
                sorted_distance[:k]
            )

        return self.__vote(nearst_neighbors)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = sum(y_pred == y_test) / len(y_test)
        return accuracy

    def __vote(self, nearst_neighbors):
        return max(set(nearst_neighbors), key=nearst_neighbors.count)

    def __euclidean_distance(self, point_a, points_b):
        distances = []
        for point in points_b:
            distances.append(np.sqrt(np.sum(np.square(
                    point_a - point))))
        return distances