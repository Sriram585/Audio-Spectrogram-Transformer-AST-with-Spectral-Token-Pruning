import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
import scipy.io.wavfile as wav
import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset

ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"

def download_esc50(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    extracted_dir = os.path.join(data_dir, "ESC-50-master")
    if os.path.exists(extracted_dir):
        return extracted_dir

    zip_path = os.path.join(data_dir, "ESC-50.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading ESC-50 dataset from {ESC50_URL} (~600MB)...")
        urllib.request.urlretrieve(ESC50_URL, zip_path)
    print("Extracting ESC-50 archive...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_dir)
    return extracted_dir

class ESC50SpectrogramDataset(Dataset):
    """
    Loads 5-second .wav clips, resamples to 16kHz, and converts 1D waveforms
    into normalized 2D Log-Mel Spectrograms of shape [1, 64, 200].
    """
    def __init__(self, root_dir, folds=(1, 2, 3, 4), esc10_only=True):
        meta_path = os.path.join(root_dir, "meta", "esc50.csv")
        df = pd.read_csv(meta_path)
        if esc10_only:
            df = df[df["esc10"] == True].copy()
        df = df[df["fold"].isin(folds)].reset_index(drop=True)

        unique_cats = sorted(df["category"].unique())
        self.label_map = {cat: idx for idx, cat in enumerate(unique_cats)}
        self.df = df
        self.audio_dir = os.path.join(root_dir, "audio")

        self.mel_spec = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=400,
            n_mels=64
        )
        self.amp_to_db = T.AmplitudeToDB(top_db=80)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = os.path.join(self.audio_dir, row["filename"])
        sr, data = wav.read(wav_path)

        # Convert int16 PCM to float32 [-1.0, 1.0]
        if data.dtype != np.float32:
            data = data.astype(np.float32) / (np.max(np.abs(data)) + 1e-6)

        waveform = torch.from_numpy(data).unsqueeze(0)  # [1, Samples]

        # Fast downsample from 44.1kHz to ~16kHz (take every 3rd sample -> ~14.7kHz, pad/crop to 80,000 samples)
        if sr > 16000:
            step = max(1, int(round(sr / 16000)))
            waveform = waveform[:, ::step]

        target_len = 16000 * 5  # 80,000 samples (5 seconds)
        if waveform.shape[1] < target_len:
            waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[1]))
        else:
            waveform = waveform[:, :target_len]

        # Convert to 2D Log-Mel Spectrogram: [1, 64, 201] -> crop to [1, 64, 200]
        spec = self.amp_to_db(self.mel_spec(waveform))[:, :, :200]

        # Instance-level spectral subtraction (cancels steady background hum)
        spec = (spec - spec.mean(dim=-1, keepdim=True)) / (spec.std() + 1e-6)

        label = self.label_map[row["category"]]
        return spec, torch.tensor(label, dtype=torch.long)