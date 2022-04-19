import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randInt
from sklearn.pipeline import Pipeline

def read_dataset(dst_path):
  dataset = glob.glob(dst_path)
  return dataset

def perform_min_max_scaling(vec):
  return (vec - vec.min()) / (vec.max() - vec.min())
  
def read_csv(dataset):
  batch = np.zeros((2, 20, 1903), dtype=np.float64)
  gt_labels = []
  
  for i, file_path in enumerate(dataset):
    gt_labels.append(file_path.split("/")[3])
    df = pd.read_csv(file_path)
    
    ts_tp9 = df["TP9"].to_numpy()
    ts_tp10 = df["TP10"].to_numpy()
  
    batch[0,i,:] = ts_tp9[996:2899]
    batch[1,i,:] = ts_tp10[996:2899]
  return batch, gt_labels

  
if __name__ == "__main__":
  dataset = read_dataset("./cs598-eeg-data/data_v2/*/*")
  ts_batch, ts_gt_labels = read_csv(dataset)
  x_train, x_test, y_train, y_test = train_test_split(ts_batch[0], ts_gt_labels, test_size=0.3, random_state=42, shuffle=True)
  
  pipe = Pipeline([
    ("rf", RandomForestClassifier(random_state=0, n_estimators=10, max_depth=10)),
  ])
  parameters = {
    'rf__n_estimators' : sp_randInt(1, 50),
    'rf__max_depth'    : sp_randInt(2, 25),
  }

  search = RandomizedSearchCV(estimator=pipe,
                              n_iter=5,
                              param_distributions=parameters,
                              random_state=0)
  search.fit(x_train, y_train)
  print(search.best_params_, search.best_score_)
  model = search
  
  y_pred = model.predict(x_test)
  print(classification_report(y_test, y_pred, target_names=["turn_on", "turn_off"]))
  
  tp,tn,fp,fn = 0,0,0,0
  
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