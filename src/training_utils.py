import os
import random
from pathlib import Path
import numpy as np
from src.common import write_json


def set_seed(seed, cpu=False, threads=None):
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf
    if threads:
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def callbacks(directory, best=None, phase="head"):
    from tensorflow import keras
    import time
    directory = Path(directory)
    checkpoint = keras.callbacks.ModelCheckpoint(str(directory / "best.keras"), monitor="val_loss", save_best_only=True, verbose=1)
    if best is not None:
        checkpoint.best = best
    class Progress(keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.began = time.perf_counter()

        def on_train_batch_end(self, batch, logs=None):
            if batch == 0 or (batch + 1) % 100 == 0:
                elapsed = time.perf_counter() - self.began
                print(f"Training batch {batch + 1}/{self.params.get('steps', '?')}; {elapsed:.1f}s elapsed", flush=True)

        def on_epoch_end(self, epoch, logs=None):
            if logs is not None:
                logs["epoch_seconds"] = time.perf_counter() - self.began

    return [Progress(), checkpoint,
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=.5, patience=2, min_lr=1e-7),
            keras.callbacks.CSVLogger(str(directory / f"{phase}_epochs.csv")),
            keras.callbacks.TerminateOnNaN()]


def save_history(directory, histories):
    combined = {}
    for history in histories:
        for key, values in history.history.items():
            combined.setdefault(key, []).extend(float(value) for value in values)
    write_json(Path(directory) / "history.json", combined)
    return combined
