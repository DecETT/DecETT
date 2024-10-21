from typing import Any
import numpy as np
    
class DRLPacketSizeSequence:
    def __init__(self, max_len=200, max_size=1449, signed=True, padding=True):
        self.max_len = max_len
        self.signed = signed
        self.padding = padding
        self.max_size = max_size if max_size > 0 else np.iinfo(np.int32).max
        self.max_packet_len = 0
    
    def __call__(self, tls_flow, tun_flow, args) -> Any:
        return self.session_features(tls_flow, tun_flow, args)
    
    def session_features(self, tls_flow, tun_flow, args) -> np.ndarray:
        
        tls_seqs = np.array(tls_flow["seq_payload"])
        tun_seqs = np.array(tun_flow["seq_payload"])
        self.max_packet_len = max(self.max_packet_len, np.max(np.abs(tun_seqs)))
        
        dir_tls_seqs = np.sign(tls_seqs)
        dir_tun_seqs = np.sign(tun_seqs)


        label = tls_flow[args.label]

        if not args.keep_zero:
            zero_indices_tls = np.where(tls_seqs == 0)[0]
            zero_indices_tun = np.where(tun_seqs == 0)[0]

            tls_seqs = np.delete(tls_seqs, zero_indices_tls)
            tun_seqs = np.delete(tun_seqs, zero_indices_tun)
            dir_tls_seqs = np.delete(dir_tls_seqs, zero_indices_tls)
            dir_tun_seqs = np.delete(dir_tun_seqs, zero_indices_tun)


        if len(dir_tls_seqs) < args.min_num_pkts:
            return [], [], []

        x_tls, x_tun, y = [], [], []
        tls_features, tun_features = self.extractor(tls_seqs, tun_seqs, args)
        x_tls.append(tls_features)
        x_tun.append(tun_features)
        y.append(label)

        return x_tls, x_tun, y

    def extractor(self, tls_seqs:np.ndarray, tun_seqs:np.ndarray, args):
        tls_feature = tls_seqs
        tun_feature = tun_seqs
        max_len = args.max_num_pkts

        if len(tls_feature) < max_len and self.padding:
            tls_feature = np.pad(tls_feature, (0, max_len - len(tls_feature)))
        elif len(tls_feature) > max_len:
            tls_feature = tls_feature[:max_len]

        if len(tun_feature) < max_len and self.padding:
            tun_feature = np.pad(tun_feature, (0, max_len - len(tun_feature)))
        elif len(tun_feature) > max_len:
            tun_feature = tun_feature[:max_len]

        return tls_feature, tun_feature