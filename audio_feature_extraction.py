import librosa, torch, numpy
import scipy.signal as signal
import noisereduce as nr


#Feature extraction (maybe?)

y, sr = librosa.load('data/audio.wav', sr=16000)

# Band-pass filter 80–8000 Hz
sos = signal.butter(10, [80, 8000], btype='bandpass', fs=sr, output='sos')
y_filt = signal.sosfilt(sos, y)

# Take the first 1.5 seconds as the noise sample
noise_len = int(1.5 * sr)
noise_clip = y_filt[:noise_len]

# Reduce stationary MRI noise using spectral gating
y_clean = nr.reduce_noise(y=y_filt, y_noise=noise_clip, sr=sr)


mel = librosa.feature.melspectrogram(y=y_clean, sr=sr, n_mels=128)
mel_db = librosa.power_to_db(mel)
print(type(mel_db))
print(mel_db.shape)



