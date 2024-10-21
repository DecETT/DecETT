from typing import Any, Optional, Tuple
from torch.autograd import Function
import torch
import torch.nn as nn
import torch.nn.functional as F


MAX_PACKET_LEN = 3000

class BiGRU(nn.Module):
    def __init__(self, input_size=16, hidden_size=128, num_layers=2, dropout=0.2) -> None:
        super(BiGRU, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.gru_layers = nn.ModuleList()
        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size * 2
            gru_layer = nn.GRU(in_size, hidden_size, bidirectional=True, batch_first=True)
            self.gru_layers.append(gru_layer)
        
    def forward(self, inputs:torch.Tensor, hidden:torch.Tensor=None):
        outputs = [inputs]
        hiddens = []

        for layer in range(self.num_layers):
            gru_layer = self.gru_layers[layer]
            output, hidden = gru_layer(outputs[-1], hidden)
            if layer != self.num_layers - 1:
                output = F.dropout(output, p=self.dropout, training=self.training, inplace=False)
            outputs.append(output) 
            hiddens.append(hidden)

        outputs = torch.cat(outputs[1:], dim=2)
        hiddens = torch.cat(hiddens, dim=0).permute(1, 0, 2).reshape(inputs.size(0), -1)

        return outputs, hiddens


class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GRL(nn.Module):
    def __init__(self):
        super(GRL, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class Encoder_Content(nn.Module):
    
    def __init__(self, emb_size, hidden_size, num_layers, dropout=0.2) -> None:
        super(Encoder_Content, self).__init__()

        self.encoder = BiGRU(emb_size, hidden_size, num_layers, dropout=dropout)

        self.share = nn.Sequential(
            nn.Linear(in_features = hidden_size * num_layers * 2, out_features = hidden_size * num_layers)
        )
        
        
    def forward(self, emb_tls:torch.Tensor, emb_tun:torch.Tensor):
        content_tls, content_tls_hn = self.encoder(emb_tls)
        content_tls = self.share(content_tls)

        content_tun, content_tun_hn = self.encoder(emb_tun)
        content_tun = self.share(content_tun)

        return content_tls, content_tls_hn, content_tun, content_tun_hn


class Encoder_Attr(nn.Module):
    def __init__(self, emb_size, hidden_size, num_layers, dropout=0.2):
        super(Encoder_Attr, self).__init__()
        self.encoder_x = BiGRU(emb_size, hidden_size, num_layers, dropout=dropout)
        self.linear = nn.Sequential(
            nn.Linear(in_features = hidden_size * num_layers * 2, out_features = hidden_size * num_layers)
        )
        
    def forward(self, x):
        attr_output, attr_hn = self.encoder_x(x)
        attr_output = self.linear(attr_output)

        return attr_output, attr_hn


class Decoder(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.decoder = BiGRU(hidden_size*num_layers*2*2, hidden_size, num_layers, dropout=dropout)

    def forward(self, content_hn, attr_hn):
        concat = torch.cat([content_hn, attr_hn], dim=2)

        dec_output, dec_hn_tls = self.decoder(concat)

        return dec_output, dec_hn_tls


class Classifier(nn.Module):
    def __init__(self, args, in_features, dropout=0.2) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, in_features//2), 
            nn.SELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features//2, args.class_num)
        )

    def forward(self, x):
        logits = self.classifier(x)
        return logits


class Classifier_GRL(nn.Module):
    def __init__(self, args, in_features, GRL=None, dropout=0.2) -> None:
        super().__init__()
            
        self.classifier = nn.Sequential(
            nn.Linear(in_features, in_features//2), 
            nn.SELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features//2, args.class_num)
        )
        if GRL:
            self.grl = GRL()

    def forward(self, x):
        if getattr(self, 'grl', None) is not None:
            x = self.grl(x)
        logits = self.classifier(x)
        return logits
    

class DRL(nn.Module):
    def __init__(self, args, max_packet_len=MAX_PACKET_LEN, emb_size=16, hidden_size=128, num_layers=2, dropout=0.2) -> None:
        super().__init__()
        self.max_packet_len = max_packet_len
        self.max_num_pkts = args.max_num_pkts

        self.embedding = nn.Embedding(2*max_packet_len+1, emb_size)
        self.enc_con = Encoder_Content(emb_size, hidden_size, num_layers, dropout=dropout)

        self.enc_attr_tls = Encoder_Attr(emb_size, hidden_size, num_layers, dropout=dropout)
        self.enc_attr_tun = Encoder_Attr(emb_size, hidden_size, num_layers, dropout=dropout)

        self.dec = Decoder(hidden_size, num_layers, dropout=dropout)

        self.rec_tls = nn.Sequential(
            nn.Linear(hidden_size*num_layers*2, hidden_size),
            nn.SELU(),
            nn.Linear(hidden_size, 2*max_packet_len+1)
        )
        self.rec_tun = nn.Sequential(
            nn.Linear(hidden_size*num_layers*2, hidden_size),
            nn.SELU(),
            nn.Linear(hidden_size, 2*max_packet_len+1)
        )

        self.dense = nn.Sequential(
            nn.Linear(hidden_size*num_layers*2*2, hidden_size*2),
            nn.SELU(),
            nn.Dropout(dropout)
        )
        self.classifier = Classifier(args, hidden_size * num_layers, dropout=dropout)
        self.classifier_reverse_app = Classifier_GRL(args, hidden_size * num_layers, GRL=GRL, dropout=dropout)
        
    
    def forward(self, X):

        X = X.to(torch.long)

        x_tls = X[:, 0, :]
        x_tun = X[:, 1, :]
        
        emb_tls = self.embedding(x_tls + abs(self.max_packet_len))
        emb_tun = self.embedding(x_tun + abs(self.max_packet_len))

        _, con_tls_hn, _, con_tun_hn = self.enc_con.forward(emb_tls, emb_tun)

        features_tls = torch.cat([con_tls_hn, con_tls_hn], dim=1)
        features_tun = torch.cat([con_tun_hn, con_tun_hn], dim=1)

        features_tls = self.dense(features_tls)
        features_tun = self.dense(features_tun)
        
        pred_tls = self.classifier(features_tls)
        pred_tun = self.classifier(features_tun)

        
        return pred_tls, pred_tun