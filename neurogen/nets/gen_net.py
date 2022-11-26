import math
import torch
from torch import Tensor
from torch import nn
from torch.nn import (
    TransformerEncoder, 
    TransformerEncoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
)
from utils.nn_utils import generate_square_subsequent_mask


    
class PosEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


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
        self._gen_mask = params['gen_mask']
        
        self._d_model = d_model
        self._dilation_depth = dilation_depth + 1 # +initial with no dilation

        # Define positional encoder
        self._pos_encoder = PosEncoding(d_model, dropout)
        
        # Define encoders
        self._dilated_encoders = nn.ModuleList([])
        for _ in range(self._dilation_depth):
            enc_layer = TransformerEncoderLayer(d_model, 
                                                n_heads, 
                                                d_hidden,
                                                dropout,)
            encoder = TransformerEncoder(enc_layer, n_enc_layers)
            self._dilated_encoders.append(encoder)
        
        # Define decoders
        self._dilated_decoders = nn.ModuleList([])
        for _ in range(self._dilation_depth):
            dec_layer = TransformerDecoderLayer(d_model, 
                                                n_heads, 
                                                d_hidden,
                                                dropout,)
            decoder = TransformerDecoder(dec_layer, n_dec_layers)
            self._dilated_decoders.append(decoder) 
        
        # Last FC decoder
        self._last_fc = nn.Linear(d_model, n_q_out)
        self._softmax = nn.Softmax(dim=-1)
        
        # Init weights
        #self._init_weights()  
        
    def _init_weights(self) -> None:
        initrange = 0.1
        self._last_fc.bias.data.zero_()
        self._last_fc.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, x):
        """
            x: Tensor, shape [batch_size, dilation_depth, seq_len, embedding_dim]
        """
        # Make [dilation_depth, seq_len, batch_size, embedding_dim]
        x = torch.permute(x, (1, 2, 0, 3))#.contiguous() 
        x_device = x.get_device()	
        dec_out = x[0]
        if self._gen_mask:
            enc_mask = generate_square_subsequent_mask(x.shape[1]).to(x_device)
        else:
            enc_out = None
        for ddx in range(self._dilation_depth):
            x_d = x[ddx]
            x_d_pos = self._pos_encoder(x_d * math.sqrt(self._d_model))
            enc_out = self._dilated_encoders[ddx](x_d_pos, enc_mask)
            if ddx == 0:
                dec_out = self._dilated_decoders[ddx](enc_out, enc_out)
            else:
                dec_out = self._dilated_decoders[ddx](dec_out, enc_out)
        fc_out = self._last_fc(dec_out)
        sm_out = self._softmax(fc_out)
        return fc_out
        
            
class SimpleTMonoSampler(nn.Module):

    def __init__(self,
                 params, 
                 ) -> None:
        super().__init__()
        
        # Unpack model parameters
        n_q_out = params['n_q_out']
        d_model = params['d_model']
        d_hidden = params['d_hidden']
        n_heads = params['n_heads']
        n_layers = params['n_layers']
        dropout = params['dropout']
        
        self._d_model = d_model

        # Define positional encoder
        self._pos_encoder = PosEncoding(d_model, dropout)
        
        # Define encoders
        enc_layer = TransformerEncoderLayer(d_model, 
                                            n_heads, 
                                            d_hidden,
                                            dropout,)
        self._encoder = TransformerEncoder(enc_layer, n_layers)
        
        # Last FC decoder
        self._last_fc = nn.Linear(d_model, n_q_out)
        self._softmax = nn.Softmax(dim=-1)
        
        # Init weights
        #self._init_weights()  
        
    def _init_weights(self) -> None:
        initrange = 0.1
        self._last_fc.bias.data.zero_()
        self._last_fc.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, x):
        """
            x: Tensor, shape [batch_size, dilation_depth, seq_len, embedding_dim]
        """
        # Make [dilation_depth, seq_len, batch_size, embedding_dim]
        x = torch.permute(x, (1, 2, 0, 3))#.contiguous() 
        x_pos = self._pos_encoder(x[0] * math.sqrt(self._d_model))
        enc_out = self._encoder(x_pos)
        fc_out = self._last_fc(enc_out)
        sm_out = self._softmax(fc_out)
        return fc_out

        
