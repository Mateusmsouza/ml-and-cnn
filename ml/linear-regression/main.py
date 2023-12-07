from linear import LinearRegression
from sklearn.linear_model import LinearRegression as LinearRegressionSklearn
import pandas as pd


def get_dataset():
    data = pd.read_csv('../../datasets/starbucks.csv')
    data = data.sample(frac=1).reset_index(drop=True)
    training_test_index = round(len(data) * 0.8)
    
    training_data = data[:training_test_index].reset_index(drop=True)
    testing_data = data[training_test_index:].reset_index(drop=True)
    return  training_data, testing_data

if __name__ == '__main__':
    training_data, testing_data = get_dataset()
    linear_regresson = LinearRegression()
    linear_regresson_sk = LinearRegressionSklearn()
    
    linear_regresson.fit(training_data['Calories'], training_data[' Total Carbohydrates (g) '])
    score = linear_regresson.score(testing_data['Calories'], testing_data[' Total Carbohydrates (g) '])

    linear_regresson_sk.fit(training_data['Calories'].values.reshape(-1, 1), training_data[' Total Carbohydrates (g) '].values.reshape(-1, 1))
    score_sk = linear_regresson_sk.score(testing_data['Calories'].values.reshape(-1, 1), testing_data[' Total Carbohydrates (g) '].values.reshape(-1, 1))
    print(f'score sklearn: {score_sk}, score ours: {score}')