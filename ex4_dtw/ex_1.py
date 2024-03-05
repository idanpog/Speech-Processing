import os
import numpy as np
import librosa
from tqdm import tqdm
 
# define the euclidean distance
def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


# implement the dynamic time warping algorithm
def dynamic_time_warping(x, y):
    dtw_matrix = np.zeros((len(x) + 1, len(y) + 1))
    dtw_matrix[0, 1:] = np.inf
    dtw_matrix[1:, 0] = np.inf
    
    for i, x_t in enumerate(x, start=1):
        for j, y_t in enumerate(y, start=1):
            dtw_matrix[i, j] = euclidean_distance(x_t, y_t) + \
            min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])

    return dtw_matrix[-1, -1]


# implement the k-nearest neighbors algorithm
class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X, metric="euclidean"):
        distances = np.zeros((X.shape[0], self.X.shape[0]))
        for i, x in enumerate(tqdm(X, desc = f"calculating {metric} distance", leave=False)):
            for j, x_ in enumerate(self.X):
                distances[i, j] = euclidean_distance(x, x_) if metric == "euclidean" \
                    else dynamic_time_warping(x, x_)

        y_pred = np.zeros(X.shape[0])
        for i, x in enumerate(tqdm(X, desc = f"predicting based on {metric} distance", leave=False)):
            nearest_neighbors = np.argsort(distances[i])[:self.k]
            nearest_labels = self.y[nearest_neighbors]
            y_pred[i] = np.bincount(nearest_labels).argmax()
        return y_pred


def numeric_from_str(label):
    return {
        "one": 0, 
        "two": 1,
        "three": 2,
        "four": 3,
        "five": 4}[label]

def load_features(f_path):
    y, sr = librosa.load(f_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    return mfcc.T

def load_train(path):
    x, y = [], []
    for label in "one", "two", "three", "four", "five":
        label_path = f"{path}/{label}"
        for filename in os.listdir(label_path):
            if filename.endswith("wav"):
                y.append(numeric_from_str(label))
                x.append(load_features(f"{label_path}/{filename}"))
    return np.array(x), np.array(y)


def get_sample_number(filename):
    return int(filename.split(".")[0].replace("sample", ''))

def load_test(path):
    x, names = [], []
    test_files = [f for f in os.listdir(path) if f.endswith("wav")]
    test_files = sorted(test_files, key=get_sample_number)
    for filename in test_files:
            x.append(load_features(f"{path}/{filename}"))
            names.append(filename)
    return np.array(x), names

def main():
    train_path = 'train_data'
    test_path = 'test_files'
    x_train, y_train = load_train(train_path)
    x_test, y_names = load_test(test_path)
    knn = KNN(k=1)
    knn.fit(x_train, y_train)
    y_pred_dtw = knn.predict(x_test, metric="dtw")
    y_pred_euclidean = knn.predict(x_test, metric="euclidean")
    with open("output.txt", "w") as f:
        for name, pred_dtw, pred_euclidean in zip(y_names, y_pred_dtw, y_pred_euclidean):
            f.write(f"{name} - {int(pred_euclidean) + 1} - {int(pred_dtw) + 1}\n")
if __name__ == "__main__":
    main()


        