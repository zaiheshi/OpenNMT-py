"""Define RNN-based encoders."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

# from onmt.utils.rnn_factory import rnn_factory


def is_decreasing(lst):
    if len(lst) == 1:
        return True
    p = lst[0]
    for x in lst[1:]:
        if x > p:
            return False
        p = x
    return True

class Separator(nn.Module):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (str):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :class:`torch.nn.Dropout`
       embeddings (onmt.modules.Embeddings): embedding module to use
    """

    def __init__(self, hidden_size=1024, dropout=0.0, temperature = 1, embeddings=None):
        super(Separator, self).__init__()
        assert embeddings is not None

        num_directions = 2
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings
        self.temperature = temperature
        self.no_pack_padded_seq = False
        self.bi_lstm = nn.LSTM( input_size=embeddings.embedding_size,
                                hidden_size=hidden_size,
                                num_layers=1,
                                dropout=dropout,
                                bidirectional=True)
        self.linear = nn.Linear(hidden_size*num_directions, 2)
        # self.lstm = nn.LSTM( input_size=embeddings.embedding_size,
        #                         hidden_size=hidden_size,
        #                         num_layers=1,
        #                         dropout=dropout,
        #                         bidirectional=False)


    def forward(self, src, lengths=None):
        # src: seq * batch * 1
        # length: batch
        
        emb = self.embeddings(src)
        # s_len, batch, emb_dim = emb.size()

        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Tensor.
            # 此处判断lengths_list是否是降序排列
            lengths_list = lengths.view(-1).tolist()
            if not is_decreasing(lengths_list):
                raise RuntimeError("separator.py, is_decreasing() = False")
            packed_emb = pack(emb, lengths_list)
        else:
            raise RuntimeError("separator.py, rnn pack error")

        memory_bank, encoder_final = self.bi_lstm(packed_emb)

        if lengths is not None and not self.no_pack_padded_seq:
            memory_bank = unpack(memory_bank)[0]
        # seq * batch * 2
        output = F.gumbel_softmax(self.linear(memory_bank), tau = self.temperature, hard = False)
        # return encoder_final, memory_bank, lengths
        
        #     emb , P(z) = 1 
        return emb, output[:,:,0] 

    # def update_dropout(self, dropout):
    #    self.rnn.dropout = dropout



if __name__ == "__main__":
    embedding = nn.Embedding(15,16)
    embedding.embedding_size = 16
    model = Separator(32, 0.0, 1, embedding)
    input = torch.arange(12).reshape(3,4)
    lengths = torch.tensor([3,3,3,3])
    # 3*4*32, ((4, 32), (4, 32)),4 
    res = model(input, lengths)
    # print(res.size())