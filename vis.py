from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randInt
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import joblib
import glob
import pandas as pd
import numpy as np


def read_dataset(dst_path):
    dataset = glob.glob(dst_path)
    return dataset


def plot_confusion_matrix_custom(cm, classes, normalize=False, title='Confusion matrix', cmap='PuBu'):
    import itertools
    plt.figure(figsize=[5, 4])
    plt.grid(False)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)

    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def min_max_scaling(vec):
    return (vec - vec.min()) / (vec.max() - vec.min())


def read_csv(dataset):
    batch = np.zeros((4, 50, 1903), dtype=np.float64)
    gt_labels = []

    for i, file_path in enumerate(dataset):
        gt_labels.append(file_path.split("/")[3])
        df = pd.read_csv(file_path)

        ts_tp9 = df["TP9"].to_numpy()
        ts_tp10 = df["TP10"].to_numpy()
        ts_af7 = df["AF7"].to_numpy()
        ts_af8 = df["AF8"].to_numpy()

        batch[0, i, :] = ts_tp9[996:2899]
        batch[1, i, :] = ts_tp10[996:2899]
        batch[2, i, :] = ts_af7[996:2899]
        batch[3, i, :] = ts_af8[996:2899]
    return batch, gt_labels


def train_model(x_train, y_train):
    pipe = Pipeline([
        ("rf", RandomForestClassifier(random_state=0, n_estimators=10, max_depth=10)),
    ])
    parameters = {
        'rf__n_estimators': sp_randInt(1, 100),
        'rf__max_depth': sp_randInt(2, 100),
    }

    search = RandomizedSearchCV(estimator=pipe,
                                n_iter=100,
                                cv=10,
                                param_distributions=parameters,
                                random_state=0)
    search.fit(x_train, y_train)
    print(search.best_params_, search.best_score_)
    model = search

    joblib.dump(
        model, './models/model_RF_{}_{}.pkl'.format(search.best_params_, search.best_score_))
    return model


def test_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred,
          target_names=["turn_on", "turn_off"]))
    return y_pred


def get_metrics(y_test, y_pred):
    tp, tn, fp, fn = 0, 0, 0, 0

    for pred, gt in zip(y_pred, y_test):
        if (pred == "turn_on" and gt == "turn_on"):
            tp += 1
        elif pred == "turn_off" and gt == "turn_off":
            tn += 1
        elif pred == "turn_off" and gt == "turn_on":
            fn += 1
        elif pred == "turn_on" and gt == "turn_off":
            fp += 1

    precision = tp/(tp+fp+1e-8)
    recall = tp/(tp+fn+1e-8)
    accuracy = (tp+tn)/(tp+tn+fp+fn+1e-8)
    f1_score = (2*precision*recall)/(recall+precision+1e-8)

    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("Accuracy: {}".format(accuracy))
    print("F-1 score: {}".format(f1_score))


if __name__ == "__main__":
    dataset = read_dataset("./cs598-eeg-data/data_v2/*/*")
    ts_batch, ts_gt_labels = read_csv(dataset)
    x_train, x_test, y_train, y_test = train_test_split(np.column_stack(
        (ts_batch[0], ts_batch[1], ts_batch[2], ts_batch[3])), ts_gt_labels, test_size=0.3, random_state=42, shuffle=True)

    model = train_model(x_train, y_train)

    # model = joblib.load(
    #     "./models/model_RF_{'rf__max_depth': 38, 'rf__n_estimators': 88}_0.8833333333333332.pkl")
    y_pred = test_model(model, x_test, y_test)
    get_metrics(y_test, y_pred)

    confusion_matrix_output = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix_custom(confusion_matrix_output, classes=[
                                 'turn_on', 'turn_off'], title='Confusion matrix')
