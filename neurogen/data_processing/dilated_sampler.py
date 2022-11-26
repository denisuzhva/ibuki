import torch



def dil_sampler_recept_field(patch_size, n_patches, dilation_depth):
    max_dil_size = 2**dilation_depth
    recept_field = patch_size * (max_dil_size * (n_patches - 1) + 1)
    return recept_field

    
def dilated_sampler_wav(in_wav, 
                        sample_index, 
                        patch_size, 
                        n_patches, 
                        dilation_depth,
                        receptive_field=None):
    device = in_wav.get_device()
    patched_seq = torch.zeros((dilation_depth + 1,  # number of dilations + no dilation case
                               n_patches,           # sequence length
                               patch_size,          # patch size (embedding dim?)
                               ))
    if not receptive_field:
        receptive_field = dil_sampler_recept_field(patch_size, n_patches, dilation_depth)
    recept_data = in_wav[sample_index : sample_index + receptive_field]
    recept_unfolded = recept_data.unfold(-1, patch_size, patch_size)
    for dil_deg in range(dilation_depth + 1):
        dil = 2 ** dil_deg
        recept_selected =  recept_unfolded[::dil, :][-n_patches:]
        patched_seq[dil_deg, :, :] = recept_selected
    return patched_seq

    
def dilated_sampler_sg(in_mel, 
                       sample_index, 
                       sample_size, 
                       seq_len, 
                       dilation_depth,
                       receptive_field=None):
    device = in_mel.get_device()
    patched_seq = torch.zeros((dilation_depth + 1,  # number of dilations + no dilation case
                               seq_len,             # sequence length
                               sample_size,         # size of a sample (dimensionality, e.g. # of mels) 
                               ))
    if not receptive_field:
        receptive_field = dil_sampler_recept_field(1, seq_len, dilation_depth)
    recept_data = in_mel[sample_index : sample_index + receptive_field]
    for dil_deg in range(dilation_depth + 1):
        dil = 2 ** dil_deg
        recept_selected =  recept_data[::dil][-seq_len:]
        patched_seq[dil_deg, :, :] = recept_selected
    return patched_seq

    
