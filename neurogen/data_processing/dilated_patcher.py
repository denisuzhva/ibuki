import torch



def dil_patcher_recept_field(patch_size, n_patches, dilation_depth):
    max_dil_size = 2**dilation_depth
    recept_field = patch_size * (max_dil_size * (n_patches - 1) + 1)
    return recept_field

    
def dilated_patcher(in_seq, sample_index, patch_size, n_patches, dilation_depth,):
    device = in_seq.get_device()
    patched_seq = torch.zeros((dilation_depth + 1,  # number of dilations + no dilation case
                               n_patches,           # sequence length
                               patch_size,          # patch size (embedding dim?)
                               ))
    receptive_field = dil_patcher_recept_field(patch_size, n_patches, dilation_depth)
    recept_data = in_seq[sample_index : sample_index + receptive_field]
    recept_unfolded = recept_data.unfold(-1, patch_size, patch_size)
    for dil_deg in range(dilation_depth + 1):
        dil = 2 ** dil_deg
        recept_selected =  recept_unfolded[::dil, :][-n_patches:]
        patched_seq[dil_deg, :, :] = recept_selected
    return patched_seq