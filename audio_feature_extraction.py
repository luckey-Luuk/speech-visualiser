import librosa, torch, numpy




#Feature extraction (maybe?)
y, sr = librosa.load('data/audio.wav', sr=16000)
mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
mel_db = librosa.power_to_db(mel)
print(type(mel_db))
print(mel_db.shape)

