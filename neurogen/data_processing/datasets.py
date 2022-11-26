import torch as t
import torch.nn.functional as F
import torchaudio as ta
#from librosa import load as lload
from torch.utils.data import Dataset
from data_processing.dilated_sampler import (
    dil_sampler_recept_field,
    dilated_sampler_wav,
    dilated_sampler_sg,
)



class SimpleWavHandler(Dataset):

    def __init__(self, params,) -> None:
        super().__init__()
        
        path_to_wav = params['path_to_wav']
        mono = params['mono']
        use_part = params['use_part']
        
        sr = params['sr']
        sample_size = params['sample_size'] 
        unfolding_step = params['uf_step']
        
        wav_data, sr_init = ta.load(path_to_wav)
        song_length = wav_data.shape[-1]
        resampler = ta.transforms.Resample(sr_init, sr)
        wav_data = resampler(wav_data)

        if mono:
            wav_data = wav_data[:int(song_length * use_part)]
            self._wav_data_uf = wav_data.unfold(0, 
                                                sample_size, 
                                                unfolding_step).view(-1, 1, sample_size),#.to(device)
        else:
            wav_data = wav_data[:, :int(song_length * use_part)]
            self._wav_data_uf = t.swapaxes(wav_data.unfold(1, 
                                                         sample_size, 
                                                         unfolding_step),#.to(device),
                                           0, 1)
        data_shape = self._wav_data_uf.shape 
        self.__n_samples = data_shape[0]
    
    def __len__(self) -> int:
        return self.__n_samples
    
    def __getitem__(self, index):
        return self._wav_data_uf[index]

        
class SimpleDilatedWavHandler(Dataset):
    
    def __init__(self, params,) -> None:
        super().__init__()
        
        path_to_wav = params['path_to_wav']
        mono = params['mono']
        pad_seq = params['pad_seq']
        use_part = params['use_part']
        
        self._sr = params['sr']
        self._patch_size = params['sample_size'] 
        self._n_patches = params['seq_len']
        self._dil_depth = params['dilation_depth']
        
        wav_data, sr_init = ta.load(path_to_wav)
        song_length = wav_data.shape[-1]
        resampler = ta.transforms.Resample(sr_init, self._sr)
        wav_data = resampler(wav_data)
       
        if mono:
            wav_data = wav_data[:int(song_length * use_part)]
            self._max_receptive_field = dil_sampler_recept_field(self._patch_size, 
                                                                 self._n_patches, 
                                                                 self._dil_depth)
            if pad_seq:
                self._wav_data = F.pad(wav_data, (self._max_receptive_field, 1), 'constant', 0)
                self.__n_samples = self._wav_data.shape[-1] - 1
            else:
                self._wav_data = F.pad(wav_data, (0, 1), 'constant', 0)
                self.__n_samples = self._wav_data.shape[-1] - self._max_receptive_field - 1
        else:
            raise NotImplementedError("Stereo data not implemented yet :C")
        
        self._wav_data *= 2**15 # to 16 bit
        self._wav_data = self._wav_data.type(t.int16)
        
    def get_context_size(self):
        in_seconds = self._max_receptive_field / self._sr
        return self._max_receptive_field, in_seconds
        
    def __len__(self) -> int:
        return self.__n_samples
    
    def __getitem__(self, sample_index):
        patched_seq = dilated_sampler_wav(self._wav_data, sample_index,
                                          self._patch_size, self._n_patches,
                                          self._dil_depth)
        target = self._wav_data[sample_index + self._max_receptive_field]
        
        return patched_seq, target#, recepdata


class SimpleDilatedMelHandler(Dataset):
    
    def __init__(self, params,) -> None:
        super().__init__()
        
        path_to_wav = params['path_to_wav']
        mono = params['mono']
        pad_seq = params['pad_seq']
        use_part = params['use_part']
        n_fft = params['n_fft']
        
        self._sr = params['sr']
        self._mel_size = params['sample_size'] 
        self._seq_len = params['seq_len']
        self._dil_depth = params['dilation_depth']
        self._hop_size = n_fft//2
        
        wav_data, sr_init = ta.load(path_to_wav)
        song_length = wav_data.shape[-1]
        resampler = ta.transforms.Resample(sr_init, self._sr)
        mel_transform = ta.transforms.MelSpectrogram(self._sr, 
                                                     n_fft, 
                                                     hop_length=self._hop_size,
                                                     n_mels=self._mel_size)
        wav_data = resampler(wav_data)

        if mono:
            wav_data = self._to_mono(wav_data)
            wav_data = wav_data[:int(song_length * use_part)]
            self._max_receptive_field = dil_sampler_recept_field(1, self._seq_len, self._dil_depth)
            mel_data = mel_transform(wav_data).T
            if pad_seq:
                self._mel_data = F.pad(mel_data, (0, 0, self._max_receptive_field, 1), 'constant', 0)
                self.__n_samples = self._wav_data.shape[0] - 1
            else:
                self._mel_data = F.pad(mel_data, (0, 0, 0, 1), 'constant', 0)
                self.__n_samples = self._mel_data.shape[0] - self._max_receptive_field - 1
        else:
            raise NotImplementedError("Stereo data not implemented yet :C")
        self._mel_data = 20 * t.log(self._mel_data + 1)
        self._mel_data /= self._mel_data.max()
        
        
    def _to_mono(self, wav_data, keepdim=False):
        return t.mean(wav_data, dim=0, keepdim=keepdim)
        
    def get_context_size(self):
        in_seconds = self._hop_size * self._max_receptive_field / self._sr
        return self._max_receptive_field, in_seconds
        
    def __len__(self) -> int:
        return self.__n_samples
    
    def __getitem__(self, sample_index):
        dilated_mel = dilated_sampler_sg(self._mel_data, sample_index,
                                         self._mel_size, self._seq_len,
                                         self._dil_depth,
                                         self._max_receptive_field)
        target = self._mel_data[sample_index + self._max_receptive_field]
        
        return dilated_mel, target#, recept_data
