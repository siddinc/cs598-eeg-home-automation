from pylsl import StreamInlet, resolve_stream, resolve_byprop
import time
import numpy as np
import joblib
import os


BUFFER_LENGTH = 25
WINDOW_LENGTH = 1903
#EPOCH_LENGTH = 1
#OVERLAP_LENGTH = 0.8
SHIFT_LENGTH = 0.4 
model = joblib.load(os.path.abspath("./model_RF.pkl"))

def setup_stream() -> list:
    try:
        print('Looking for an EEG stream')
        streams = resolve_byprop('type', 'EEG', timeout=10)

        if len(streams) == 0:
            raise RuntimeError('Can\'t find EEG stream.')
        return streams
    except RuntimeError as e:
        print(e)
    
    
def get_eeg_prediction(streams: list, n: int = 10) -> bool:
    try:
        print("Start acquiring data from EEG stream")
        inlet = StreamInlet(streams[0], max_chunklen=50)
        eeg_time_correction = inlet.time_correction()
        info = inlet.info()
        fs = int(info.nominal_srate())
        eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 4), dtype=np.float32)

        start_time = time.time()
        time_delta = 0.
        i = last_chunk_len = 0
        last_idx = 0
        
        while time_delta <= n:
            eeg_data, timestamp = inlet.pull_chunk(timeout=5, max_samples=int(fs * SHIFT_LENGTH))
            channels_data = np.array(eeg_data, dtype=np.float32)[:,0:4]
            eeg_buffer[i:i+channels_data.shape[0],:] = channels_data
            last_idx = i
            last_chunk_len = channels_data.shape[0]
            i += channels_data.shape[0]
            time_delta = time.time() - start_time

        ts = eeg_buffer[last_idx+last_chunk_len-WINDOW_LENGTH:last_idx+last_chunk_len,:]
        x_test = np.expand_dims(ts.ravel(), axis=0)
        y_pred = model.predict(x_test)
        return True if y_pred[0] == "turn_on" else False
    
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        print('Closing')
        
if __name__ == "__main__":
    streams = setup_stream()
    
    while True:
        res = get_eeg_prediction(streams)
        print(res)
