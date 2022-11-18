import torch
from torch import nn
from torch.nn import (
    TransformerEncoder, 
    TransformerEncoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
)
from neurowhore.nets.utils_dl import (
    PosEncoding,
)



class DilatedTMonoSampler(nn.Module):
    
    def __init__(self,
                 params, 
                 ) -> None:
        super().__init__()
        
        # Unpack model parameters
        n_q_out = params['n_q_out']
        d_model = params['d_model']
        d_hidden = params['d_hidden']
        n_heads = params['n_heads']
        n_enc_layers = params['n_enc_layers']
        n_dec_layers = params['n_dec_layers']
        dilation_depth = params['dilation_depth']
        dropout = params['dropout']
        
        self._dilation_depth = dilation_depth

        # Define positional encoder
        self._pos_encoder = PosEncoding(d_model, dropout)
        
        # Define encoders
        self._dilated_encoders = nn.ModuleList([])
        for _ in range(dilation_depth):
            enc_layer = TransformerEncoderLayer(d_model, 
                                                n_heads, 
                                                d_hidden,
                                                dropout,)
            encoder = TransformerEncoder(enc_layer, n_enc_layers)
            self._dilated_encoders.append(encoder)
        
        # Define decoders
        self._dilated_decoders = nn.ModuleList([])
        for _ in range(dilation_depth):
            dec_layer = TransformerDecoderLayer(d_model, 
                                                n_heads, 
                                                d_hidden,
                                                dropout,)
            decoder = TransformerEncoder(dec_layer, n_dec_layers)
            self._dilated_decoders.append(decoder) 
        
        # Last FC decoder
        self._last_fc = nn.Linear(d_model, n_q_out)
        self._softmax = nn.Softmax(dim=-1)
        
        # Init weights
        self._init_weights()  
        
    def _init_weights(self) -> None:
        initrange = 0.1
        self._last_fc.bias.data.zero_()
        self._last_fc.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, x):
        """
            x: Tensor, shape [dilation_depth, seq_len, batch_size, embedding_dim]
        """
        dec_out = x[0]
        for ddx in range(self._dilation_depth):
            x_d = x[ddx]
            enc_out = self._dilated_encoders[ddx](x_d)
            if ddx == 0:
                dec_out = self._dilated_decoders[ddx](enc_out, enc_out)
            else:
                dec_out = self._dilated_decoders[ddx](dec_out, enc_out)
        fc_out = self._last_fc(dec_out)
        return self._softmax(fc_out)
        
            