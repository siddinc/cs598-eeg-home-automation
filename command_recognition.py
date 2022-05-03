from pylsl import StreamInlet, resolve_stream, resolve_byprop
import time
import numpy as np
import joblib
import os


class CommandRecognition:
    def __init__(self, channels=4, buffer_length=25, window_length=1903, shift_length=0.4):
        self.channels = channels
        self.buffer_length = buffer_length
        self.window_length = window_length
        self.shift_length = shift_length

    def setup_stream(self) -> list:
        try:
            print('Looking for an EEG stream')
            streams = resolve_byprop('type', 'EEG', timeout=10)

            if len(streams) == 0:
                raise RuntimeError('Can\'t find EEG stream.')
            return streams
        except RuntimeError as e:
            print(e)

    def load_model(self, model_path):
        model = joblib.load(model_path)
        return model

    def get_model_prediction(self, model, streams: list, n: int = 10) -> bool:
        try:
            print("Start acquiring data from EEG stream")
            inlet = StreamInlet(streams[0], max_chunklen=50)
            eeg_time_correction = inlet.time_correction()
            info = inlet.info()
            fs = int(info.nominal_srate())
            eeg_buffer = np.zeros(
                (int(fs * self.buffer_length), self.channels), dtype=np.float32)

            start_time = time.time()
            time_delta = 0.
            i = last_chunk_len = 0
            last_idx = 0

            while time_delta <= n:
                eeg_data, _ = inlet.pull_chunk(
                    timeout=5, max_samples=int(fs * self.shift_length))
                channels_data = np.array(eeg_data, dtype=np.float32)[
                    :, 0:self.channels]
                eeg_buffer[i:i+channels_data.shape[0], :] = channels_data
                last_idx = i
                last_chunk_len = channels_data.shape[0]
                i += channels_data.shape[0]
                time_delta = time.time() - start_time

            ts = eeg_buffer[last_idx+last_chunk_len -
                            self.window_length:last_idx+last_chunk_len, :]
            x_test = np.expand_dims(ts.ravel(), axis=0)
            y_pred = model.predict(x_test)
            return True if y_pred[0] == "turn_on" else False

        except Exception as e:
            print(e)
        except KeyboardInterrupt:
            print('Closing')


if __name__ == "__main__":
    command_recognition = CommandRecognition()
    streams = command_recognition.setup_stream()
    model = command_recognition.load_model(
        os.path.abspath("./models/model_RF.pkl"))

    while True:
        res = command_recognition.get_model_prediction(model, streams)
        print(res)
