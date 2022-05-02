from cProfile import label
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv


NFFT = 256
NOVERLAP = NFFT - 1
FS = 256
CMAP = "viridis"

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

def normalize_vec(vec):
  return (vec - vec.min()) / (vec.max() - vec.min())

def plot_ts():
  # df_ton = pd.read_csv("../data_v2/turn_on/cy_1.csv")
  df_toff1 = pd.read_csv("./cs598-eeg-data/data_v3/turn_on/cy_11.csv")
  df_toff2 = pd.read_csv("./cs598-eeg-data/data_v3/turn_off/cy_11.csv")
  
  # ts_ton = df_ton["TP9"].to_numpy()
  ts_toff1 = min_max_scaling(df_toff1["TP9"].to_numpy())[996:2899]
  ts_toff2 = min_max_scaling(df_toff2["TP9"].to_numpy())[996:2899]
  
  dist = np.sqrt(np.sum(np.square(ts_toff1 - ts_toff2))) / 1934
  print(dist)
  
  # plt.plot(ts_ton, label="turn on")
  fig, ax = plt.subplots(2,1)
  
  ax[0].plot(ts_toff1, label="turn on", )
  ax[1].plot(ts_toff2, label="turn off")
  
  plt.show()
  spec1 = get_spectrogram(ts_toff1, FS, NFFT, NOVERLAP, CMAP)
  scaled_spec1 = min_max_scaling(spec1)
  # gray_spec1 = rgb2gray(scaled_spec1)
  
  spec2 = get_spectrogram(ts_toff2, FS, NFFT, NOVERLAP, CMAP)
  scaled_spec2 = min_max_scaling(spec2)
  # gray_spec2 = rgb2gray(scaled_spec2)
  
  
  plt.imshow(scaled_spec1)
  plt.show()
  
  plt.imshow(scaled_spec2)
  plt.show()



if __name__ == "__main__":
  plot_ts()