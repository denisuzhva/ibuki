import torch
from librosa import load as lload
from torch.utils.data import Dataset



class SimpleWavHandler(Dataset):

    def __init__(self, 
                 path_to_wav='',
                 sr=16000,
                 mono=True,
                 sample_size=1024,
                 unfolding_step=512,
                 use_part=1.0,
                 device=torch.device('cpu')) -> None:
        super().__init__()
        wav_data, _ = lload(path_to_wav, sr=sr, mono=mono)
        t_wav_data = torch.from_numpy(wav_data)
        song_length = t_wav_data.shape[-1]

        if mono:
            t_wav_data = t_wav_data[:int(song_length * use_part)]
            self.__t_wav_data_uf = t_wav_data.unfold(0, 
                                                     sample_size, 
                                                     unfolding_step).view(-1, 1, sample_size).to(device)
        else:
            t_wav_data = t_wav_data[:, :int(song_length * use_part)]
            self.__t_wav_data_uf = torch.swapaxes(t_wav_data.unfold(1, 
                                                                    sample_size, 
                                                                    unfolding_step).to(device),
                                                  0, 1)
        data_shape = self.__t_wav_data_uf.shape 
        self.__n_samples = data_shape[0]

    
    def __len__(self) -> int:
        return self.__n_samples
    
    def __getitem__(self, index):
        return self.__t_wav_data_uf[index]