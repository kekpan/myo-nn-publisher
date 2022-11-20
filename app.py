from collections import Counter, deque
# from datetime import datetime
from threading import Lock
from tkinter import messagebox

import myo
import numpy as np
from flask import Flask, stream_with_context
from flask_cors import CORS
from joblib import load

EMG_WINDOW = 300  # TODO: window in sec * sample rate
MAJORITY_WINDOW = 10
REST_THRESHOLD = 0.1  # TODO: threshold of signal power below which assume 'rest' gesture

# TODO: Fill with gesture names.
# Gestures[0] should be the gesture when model.predict() outputs 0. Last value should be 'rest'
Gestures = [
    '',
    'rest'
]


class EmgCollector(myo.DeviceListener):
    """
    Collects EMG data in a queue with *n* maximum number of elements.
    """

    def __init__(self, n):
        self.n = n
        self.lock = Lock()
        self.emg_data_queue = deque(maxlen=n)
        self.last_pose = 'rest'

    def get_emg_data(self):
        with self.lock:
            return list(self.emg_data_queue)

    def get_pose(self):
        with self.lock:
            return self.last_pose

    # def on_paired(self, event):
    #     ts = datetime.fromtimestamp(int(event.timestamp/1000000))
    #     print(f"[{ts}] Paired with \033[36mMyo\033[0m.")
    #     event.device.vibrate(myo.VibrationType.short)  # TODO: Need this?

    # def on_unpaired(self, event):
    #     ts = datetime.fromtimestamp(int(event.timestamp/1000000))
    #     print(f"[{ts}] Unpaired with \033[36mMyo\033[0m.")
    #     return False  # Stop the hub TODO: Need this?

    # def on_connected(self, event):
    #     ts = datetime.fromtimestamp(int(event.timestamp/1000000))
    #     print(f"[{ts}] \033[36mMyo\033[0m has connected.")
    #     event.device.stream_emg(True)  # FIXME: Temp

    # def on_disconnected(self, event):
    #     ts = datetime.fromtimestamp(int(event.timestamp/1000000))
    #     print(f"[{ts}] \033[36mMyo\033[0m has disconnected.")

    # def on_arm_synced(self, event):
    #     ts = datetime.fromtimestamp(int(event.timestamp/1000000))
    #     print(f"[{ts}] \033[36mMyo\033[0m is synced.")

    # def on_arm_unsynced(self, event):
    #     ts = datetime.fromtimestamp(int(event.timestamp/1000000))
    #     print(f"[{ts}] \033[36mMyo\033[0m is not synced.")

    # def on_unlocked(self, event): pass
    # def on_locked(self, event): pass

    def on_pose(self, event):
        try:
            if event.pose == myo.Pose.rest:
                self.last_pose = 'rest'
            elif event.pose == myo.Pose.fist:
                self.last_pose = 'fist'
            elif event.pose == myo.Pose.wave_in:
                self.last_pose = 'wave_in'
            elif event.pose == myo.Pose.wave_out:
                self.last_pose = 'wave_out'
            elif event.pose == myo.Pose.fingers_spread:
                self.last_pose = 'fingers_spread'
            elif event.pose == myo.Pose.double_tap:
                self.last_pose = 'double_tap'
        except ValueError:
            pass

    # def on_orientation(self, event): pass
    # def on_rssi(self, event): pass
    # def on_battery_level(self, event): pass

    def on_emg(self, event):
        return  # FIXME: Temp
        with self.lock:
            self.emg_data_queue.append((event.timestamp, event.emg))

    # def on_warmup_completed(self, event):
    #     ts = datetime.fromtimestamp(int(event.timestamp/1000000))
    #     print(f"[{ts}] \033[36mMyo\033[0m has warmed up.")


class Classifier(object):
    """
    Processes EMG data and outputs performed gesture.
    """

    def __init__(self, listener, filename):
        """
        Parameters
        listener: EmgCollector
            A myo.DeviceListener that listens to EMG data
        filename: str
            Path to saved classifier model with joblib, e.g. 'model.joblib'
        """
        self.listener = listener  # EmgCollector
        # Majority voting (i.e. most common gesture over the last n predictions)
        self.voting_buffer = deque(maxlen=MAJORITY_WINDOW)
        self.last_gesture = 'rest'
        if filename != None:
            # Trained classifier needs to be saved using joblib.dump()
            self.model = load(filename)
        else:
            self.model = None

    def process_emg(self, emg_data):
        pass

    def compute_features(self, emg_data):
        pass

    def get_model_prediction(self, emg_features):
        # Use argmax if model is tf/keras
        index = self.model.predict(emg_features)[0]
        return Gestures[index]

    def compute_majority_vote(self, model_output):
        self.voting_buffer.append(model_output)
        return Counter(self.voting_buffer).most_common()[0]

    def run(self):
        """ Classifier loop.
        Gets EMG data from listener, computes features, runs model and updates last_gesture
        """
        try:
            while True:
                yield ''  # For GeneratorExit purposes
                if False:  # FIXME: Temp
                    emg_data = self.listener.get_emg_data()
                    # We don't need to process the timestamps
                    emg_data = np.array([x[1] for x in emg_data]).T
                    # TODO: Check shape of emg_data: [TIME, CHANNELS] or [CHANNELS, TIME] ?

                    if emg_data.size < EMG_WINDOW * 8:
                        continue

                if self.model != None:
                    model_output = Gestures[-1]
                    if np.mean(emg_data**2) >= REST_THRESHOLD:
                        # TODO: Process data
                        emg_data = self.process_emg(emg_data)
                        # TODO: Extract features
                        emg_features = self.compute_features(emg_data)
                        # Run classifier model
                        model_output = self.get_model_prediction(emg_features)

                    # Add output to buffer for majority voting
                    gesture = self.compute_majority_vote(model_output)
                else:
                    gesture = self.listener.get_pose()

                if gesture != self.last_gesture:
                    if gesture == 'rest':
                        yield f'event: pose_off\ndata: {self.last_gesture}\n\n'
                    else:
                        yield f'event: pose\ndata: {gesture}\n\n'
                    self.last_gesture = gesture
        except GeneratorExit:
            pass


app = Flask(__name__)
CORS(app)


@app.route("/")
def generate_poses():
    # "Cache-Control": "no-store" TODO: Need this?
    return stream_with_context(classifier.run()), {"Content-Type": "text/event-stream"}


if __name__ == '__main__':
    try:
        myo.init(sdk_path='./myo-sdk-win-0.9.0/')  # myo-sdk path
        hub = myo.Hub()
        # ts = datetime.now().isoformat(sep=" ", timespec="seconds")
        # print(f"[{ts}] Attempting to find a \033[36mMyo\033[0m...")
        listener = EmgCollector(EMG_WINDOW)
        model_filename = None
        classifier = Classifier(listener, model_filename)
        with hub.run_in_background(listener.on_event):
            app.run('127.0.0.1', 8080)
    except myo._ffi.ResultError as err:
        messagebox.showerror("Myo NN Publisher", err.message.decode("utf-8"))
