import glob
import struct
import numpy as np
import librosa


def read_sbl_raw_int16(path: str, sampwidth: int = 2) -> np.ndarray:
    """
    Читает .sbl.
    Возвращает float32 сигнал примерно в диапазоне [-1, 1].
    """
    sampwidth_to_char = {1: "b", 2: "h", 4: "i"}

    with open(path, "rb") as f:
        raw_signal = f.read()

    if len(raw_signal) == 0:
        raise RuntimeError(f"Empty sbl file: {path}")

    if len(raw_signal) % sampwidth != 0:
        raw_signal = raw_signal[: len(raw_signal) - (len(raw_signal) % sampwidth)]

    num_samples = len(raw_signal) // sampwidth
    fmt = "<" + str(num_samples) + sampwidth_to_char[sampwidth]
    signal = struct.unpack(fmt, raw_signal)

    y = np.asarray(signal, dtype=np.float32)

    # нормализует под разрядность
    if sampwidth == 1:
        y /= 128.0
    elif sampwidth == 2:
        y /= 32768.0
    elif sampwidth == 4:
        y /= 2147483648.0

    return y


def main():
    frame_ms = 10.0
    n_mels = 40
    n_mfcc = 13

    data = np.load("labels_frames.npz", allow_pickle=True)
    ids = data["ids"]
    sr_list = data["sr"]
    y_list = data["y"]

    X_list = []

    for i, (file_id, sr, y_frames) in enumerate(zip(ids, sr_list, y_list)):
        if (i % 10) == 0:
            print(f"processing {i+1}/{len(ids)}", flush=True)

        sbl_paths = glob.glob(f"corpus/**/{file_id}.sbl", recursive=True)
        if not sbl_paths:
            raise RuntimeError(f"sbl not found for {file_id}")
        sbl_path = sbl_paths[0]

        y_audio = read_sbl_raw_int16(sbl_path, sampwidth=2)

        hop = int(round(sr * frame_ms / 1000.0))

        n_fft = int(4 * hop)

        # 1) log-mel спектрограмма
        mel = librosa.feature.melspectrogram(
            y=y_audio,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop,
            n_mels=n_mels,
            power=2.0,
        )
        logmel = librosa.power_to_db(mel, ref=np.max)  # (40, T)

        # 2) MFCC
        mfcc = librosa.feature.mfcc(
            y=y_audio,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop,
        )

        T = min(logmel.shape[1], mfcc.shape[1], len(y_frames))

        X = np.concatenate(
            [
                logmel[:, :T].T,  # (T, 40)
                mfcc[:, :T].T,  # (T, 13)
            ],
            axis=1,
        ).astype(
            np.float32
        )  # (T, 53)

        X_list.append(X)

    np.savez_compressed(
        "features_logmel_mfcc.npz",
        X=np.array(X_list, dtype=object),
        y=y_list,
        ids=ids,
        sr=sr_list,
        frame_ms=np.array([frame_ms]),
        feat_names=np.array(["logmel40+mfcc13"], dtype=object),
    )

    print("features_logmel_mfcc.npz")
    print(X_list[0].shape)


if __name__ == "__main__":
    main()
