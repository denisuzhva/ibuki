import torch
import torch.nn.functional as F
from librosa import load as lload
from torch.utils.data import Dataset
from data_processing.dilated_patcher import (
    dil_patcher_recept_field,
    dilated_patcher,
)



class SimpleWavHandler(Dataset):

    def __init__(self, 
                 path_to_wav='',
                 sr=16000,
                 mono=True,
                 sample_size=1024,
                 unfolding_step=512,
                 use_part=1.0,
                 #device=torch.device('cpu'),
                 ) -> None:
        super().__init__()
        wav_data, _ = lload(path_to_wav, sr=sr, mono=mono, res_type='soxr_qq')
        t_wav_data = torch.from_numpy(wav_data)
        song_length = t_wav_data.shape[-1]

        if mono:
            t_wav_data = t_wav_data[:int(song_length * use_part)]
            self._t_wav_data_uf = t_wav_data.unfold(0, 
                                                    sample_size, 
                                                    unfolding_step).view(-1, 1, sample_size),#.to(device)
        else:
            t_wav_data = t_wav_data[:, :int(song_length * use_part)]
            self._t_wav_data_uf = torch.swapaxes(t_wav_data.unfold(1, 
                                                                   sample_size, 
                                                                   unfolding_step),#.to(device),
                                                  0, 1)
        data_shape = self._t_wav_data_uf.shape 
        self.__n_samples = data_shape[0]
    
    def __len__(self) -> int:
        return self.__n_samples
    
    def __getitem__(self, index):
        return self._t_wav_data_uf[index]

        
class SimpleDilatedWavHandler(Dataset):
    
    def __init__(self,
                 path_to_wav = '',
                 sr=16000,
                 mono=True,
                 patch_size=64,
                 n_patches=64,
                 dilation_depth=5,
                 pad_wav=False,
                 use_part=1.,
                 #device=torch.device('cpu'),
                 ) -> None:
        super().__init__()
        wav_data, _ = lload(path_to_wav, sr=sr, mono=mono, res_type='soxr_qq')
        t_wav_data = torch.from_numpy(wav_data)#.to(device)
        song_length = t_wav_data.shape[-1]
        
        self._sr = sr
        self._patch_size = patch_size
        self._n_patches = n_patches
        self._dil_depth = dilation_depth

        if mono:
            t_wav_data = t_wav_data[:int(song_length * use_part)]
            max_dil_size = 2**dilation_depth
            self._max_receptive_field = dil_patcher_recept_field(patch_size, n_patches, dilation_depth)
            #self._max_receptive_field = patch_size * (max_dil_size * (n_patches - 1) + 1)
            if pad_wav:
                self._t_wav_data = F.pad(t_wav_data, (self._max_receptive_field, 1), 'constant', 0)
                self.__n_samples = self._t_wav_data.shape[-1] - 1
            else:
                self._t_wav_data = F.pad(t_wav_data, (0, 1), 'constant', 0)
                self.__n_samples = self._t_wav_data.shape[-1] - self._max_receptive_field - 1
        else:
            raise NotImplementedError("Stereo data not implemented yet :C")
        
        self._t_wav_data *= 2**15
        self._t_wav_data = self._t_wav_data.type(torch.int16)
        #self._t_wav_data = torch.arange(end = self.__n_samples)
        
    def get_context(self):
        in_seconds = self._max_receptive_field / self._sr
        return self._max_receptive_field, in_seconds
        
    def __len__(self) -> int:
        return self.__n_samples
    
    def __getitem__(self, sample_index):
        #patched_seq = torch.zeros((self._dil_depth + 1, # number of dilations + no dilation case
        #                       self._n_patches,     # sequence length
        #                       self._patch_size,    # patch size (embedding dim?)
        #                       ))
        #recept_data = self._t_wav_data[sample_index : sample_index + self._max_receptive_field]
        #recept_unfolded = recept_data.unfold(0, self._patch_size, self._patch_size)
        #for dil_deg in range(self._dil_depth + 1):
        #    dil = 2 ** dil_deg
        #    recept_selected =  recept_unfolded[::dil, :][-self._n_patches:]
        #    patched_seq[dil_deg, :, :] = recept_selected
        patched_seq = dilated_patcher(self._t_wav_data, sample_index,
                                      self._patch_size, self._n_patches,
                                      self._dil_depth)
        target = self._t_wav_data[sample_index + self._max_receptive_field]
        
        return patched_seq, target#, recept_data
