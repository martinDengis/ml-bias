from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# You can use this function in a script (located in the same folder) like this:
#
# from data import load_wine_quality
#
# X, y = load_wine_quality()

def load_wine_quality():
    """Loads and returns the (normalized) Wine Quality dataset from OpenML.

    Return
    ------
    X : array of shape (6497, 11)
        The feature matrix (input).
    y : array of shape (6497,)
        The output values vector.
    """
    dataset = fetch_openml(data_id=287, parser='auto')

    X, y = dataset.data, dataset.target
    X, y = X.to_numpy(), y.to_numpy()

    # Normalization is important for ridge regression and k-NN.
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # Shuffle the data
    X, y = shuffle(X, y, random_state=42)

    return X, y



if __name__ == "__main__":
    X, y = load_wine_quality()
    data = pd.DataFrame(X, columns=[f"Feature {i}" for i in range(X.shape[1])])
    data['Quality'] = y

    # sns.pairplot(data)

    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.show()