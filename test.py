from cProfile import label
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

def normalize_vec(vec):
  return (vec - vec.min()) / (vec.max() - vec.min())

def plot_ts():
  # df_ton = pd.read_csv("../data_v2/turn_on/cy_1.csv")
  df_toff1 = pd.read_csv("../data_v2/turn_on/cy_6.csv")
  df_toff2 = pd.read_csv("../data_v2/turn_off/cy_10.csv")
  
  # ts_ton = df_ton["TP9"].to_numpy()
  ts_toff1 = normalize_vec(df_toff1["TP9"].to_numpy()[996:2899])
  ts_toff2 = normalize_vec(df_toff2["TP9"].to_numpy()[996:2899])
  
  dist = np.sqrt(np.sum(np.square(ts_toff1 - ts_toff2))) / 1934
  print(dist)
  
  # plt.plot(ts_ton, label="turn on")
  plt.plot(ts_toff1, label="turn off1")
  plt.plot(ts_toff2, label="turn off2")
  
  plt.legend()
  plt.show()


if __name__ == "__main__":
  plot_ts()