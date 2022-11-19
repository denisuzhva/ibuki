import os
import numpy as np
import librosa



sr = 44100
mono = True
wav_path = './datasets/wav/'
npy_path = './datasets/npy/'
artists = ['wwolves']
#song_names = ['ww_set2.wav']

for artist in artists:
    wav_path_art = os.path.join(wav_path, artist)
    npy_path_art = os.path.join(npy_path, artist)
    os.makedirs(npy_path_art, exist_ok=True)
    song_names = os.listdir(wav_path_art)

    i16_bits = 16
    i16_mav = 2**i16_bits-1
    dtype = np.int16

    for sname in song_names:
        song, _ = librosa.load(os.path.join(wav_path_art, sname), 
                               sr=sr, mono=mono, res_type='soxr_qq')
        print(f"Song {sname}, len {song.size}")
        song = (song * i16_mav).astype(dtype)
        np.save(os.path.join(npy_path_art, sname.split('.')[0] + '.npy'), song)


