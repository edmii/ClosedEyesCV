import cv2
import joblib
import cv2, os
import numpy as np

from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
# import matplotlib.pyplot as plt


def openDataSet(route):
    paths = ['open','close']
    labels = []
    images = []
    target_shape = (128,128)

    for path in paths:
        pathToFile = route + '/' + path

        for filename in os.listdir(pathToFile):
            pil_image = cv2.imread(pathToFile + '/' + filename)
            gray_img = cv2.cvtColor(pil_image, cv2.COLOR_BGR2GRAY)
            img= cv2.resize(gray_img, target_shape)
            images.append(img)

            if path == 'open':
                labels.append(0)

            else:
                labels.append(1)

    images = np.array(images)
    images = np.array(images).reshape(-1, target_shape[0] * target_shape[1])
    
    return images, labels

def ApplyPCA():
    images, labels = openDataSet(r"C:\Users\micci\Desktop\closedEyeDetection-ML\dataset")
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3)


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print('applying pca')
    pca = PCA(n_components=13)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print('pca fit')

    pca_components = pca.components_
    explained_variance_ratio = pca.explained_variance_ratio_

    return X_train_pca , y_train, X_test_pca, y_test, pca_components

def model():
    x_train, y_train, x_test, y_test, components = ApplyPCA()

    model = LogisticRegression(solver='lbfgs', C=1.0, max_iter=300)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    print('Accuracy:',accuracy)
    
    joblib.dump(model, r'C:\Users\micci\Desktop\closedEyeDetection-ML\model.pkl')
    joblib.dump(components,r'C:\Users\micci\Desktop\closedEyeDetection-ML\pca_components.pkl')

model()
