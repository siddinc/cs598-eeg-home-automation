import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randInt
from sklearn.pipeline import Pipeline
from matplotlib.ticker import ScalarFormatter,AutoMinorLocator
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import cv2
import imutils
from PIL import Image

NFFT = 256
NOVERLAP = NFFT - 1
FS = 256
CMAP = "viridis"


def read_dataset(dst_path):
  dataset = glob.glob(dst_path)
  return dataset

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
  
    batch[0,i,:] = (ts_tp9[996:2899])
    batch[1,i,:] = (ts_tp10[996:2899])
    batch[2,i,:] = (ts_af7[996:2899])
    batch[3,i,:] = (ts_af8[996:2899])
  return batch, gt_labels

def get_spectrogram(data, fs, nfft, noverlap, cmap):
  fig, ax = plt.subplots(1)
  fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
  fig.dpi = 100
  ax.axis('off')
  ax.grid(False)

  pxx, freqs, bins, im = ax.specgram(x=data, Fs=fs, noverlap=noverlap, NFFT=nfft, cmap=cmap)
  return fig2rgb(fig)

def fig2rgb(fig):
  fig.canvas.draw()
  buf = fig.canvas.tostring_rgb()
  plt.close(fig)
  x = np.frombuffer(buf, dtype=np.uint8).reshape(480, 640, 3)
  return x

def rgb2gray(rgb_spec):
  r, g, b = rgb_spec[:,:,0], rgb_spec[:,:,1], rgb_spec[:,:,2]
  gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
  return gray

def min_max_scaling(vec):
  scaled_vec = (vec - vec.min()) / (vec.max() - vec.min())
  return scaled_vec

  
if __name__ == "__main__":
  dataset = read_dataset("./cs598-eeg-data/data_v3/*/*")
  ts_batch, ts_gt_labels = read_csv(dataset)
  # x_train, x_test, y_train, y_test = train_test_split(ts_batch[2], ts_gt_labels, test_size=0.3, random_state=42, shuffle=True)
  
  for i, label in zip(range(ts_batch.shape[1]), ts_gt_labels):
    specs = np.zeros((75, 100, 4))
    
    for c in range(4):
      spec = get_spectrogram(ts_batch[c,i,:], FS, NFFT, NOVERLAP, CMAP)
      scaled_spec = min_max_scaling(spec)
      resized_spec = imutils.resize(scaled_spec, height=75)
      gray_spec = rgb2gray(resized_spec)
      specs[:,:,c] = gray_spec

    np.save("/Users/elvis/Desktop/MCS/Sem2/CS598_Smart-X/final_project/cs598-eeg-data/data_v3_spec/{}/{}.npy".format(label,i+1), specs)