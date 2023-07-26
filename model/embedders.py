from typing import Tuple, List
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import numpy as np


from trainer.trainer import to_cuda
from .transformer import TransformerModel


MultiDimensionalFloat = List[float]
XYPair = Tuple[MultiDimensionalFloat, MultiDimensionalFloat]
Sequence = List[XYPair]



class Embedder(ABC, nn.Module):
    """
    Base class for embedders, transforms a sequence of pairs into a sequence of embeddings.
    """

    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def forward(self, sequences: List[Sequence]) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def batch(self, seqs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_length_after_batching(self, sequences: List[Sequence]) -> List[int]:
        pass


class ChainCoderSampleEmbedder(Embedder):
    def __init__(self, params, env):
        from .transformer import Embedding

        super().__init__()
        self.env = env
        self.params = params
        self.use_pretrained_NLP = params.use_pretrained_NLP

        self.embeddings = Embedding(
            len(self.env.i2t_inp),
            params.arch_encoder_dim,
        )
        self.pretrain_pad_emb_idx = self.env.t2i_inp['<<||SPECIAL_RESERVED_TOKEN_POSSIBLY_NEVER_USED_9||>>']
        self.activation_fn = F.relu

        self.distill_NLP_syntax_marking_ids, = to_cuda(torch.tensor(env.distill_NLP_syntax_marking_ids), use_cpu=self.params.run_on_cpu)


        self.sample_embedder = TransformerModel(
            params,
            env,
            is_encoder=True,
            with_output=False,
            use_prior_embeddings=False,
            positional_embeddings=params.enc_positional_embeddings,
            is_sample_embedder = True,
            )

        if bool(self.use_pretrained_NLP):

            self.nlp_model, self.nlp_tokenizer, self.nlp_pad_token_id = get_model_tokenizer(params.nlp_arch, 'cpu' if self.params.run_on_cpu else 'cuda')
            nlp_out_dim = 768 if params.nlp_arch=='distilbert' else 1024
            self.nlp_glue_layer = getMLP([nlp_out_dim, int(nlp_out_dim*1.5), params.arch_encoder_dim])


    def compress(
        self, tensor_iste: torch.Tensor, lens_i: torch.Tensor, 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Takes: (N_max * (d_in+d_out), B, d) tensors
        Returns: (N_max, B, d)
        """
        _i,_s,_t,_2 = tensor_iste.shape  # _2 == 2
        sb = tensor_iste.transpose(0,1).reshape(_s,-1)
        b_s = sb.transpose(0,1)
        lens_it2 = lens_i.reshape(-1,1,1).expand([_i, _t, _2]).reshape(-1)
        
        bse_out = self.sample_embedder("fwd", x=b_s, lengths=lens_it2, causal=False)

        be_out = bse_out[:,0,:]    # pooling for samples: pick first

        ite = be_out.reshape(_i, _t, _2, -1).sum(dim=-2)  # sum up content and syntax two subtokens

        return ite


    def forward(self,  tensor_ist2, samp_lens_per_inst, token_lens_per_inst, desc_inp_tensor, desc_attn_mask) -> Tuple[torch.Tensor, torch.Tensor]:

        tensor_ite = self.compress(tensor_ist2, samp_lens_per_inst)  # shape = torch.Size([158, 63, 512])

        if self.use_pretrained_NLP:  # use NLP tokenizer, mark distilled tokens with syntax subtokens, and concat distilled NLP tokens with sample_embedder output
            if self.params.fine_fune_nlp:   # in some starting epochs, it is set to False: fix params of pretrained NLP model.
                description_embs_tie = self.nlp_model(desc_inp_tensor, desc_attn_mask)
            else:
                with torch.no_grad():
                    description_embs_tie = self.nlp_model(desc_inp_tensor, desc_attn_mask)

            tensor_tie = tensor_ite.transpose(0,1)
            nlp_reserved_len = description_embs_tie.shape[0]
            _t, _i, _e = tensor_tie.shape
            syntax_markings_tie = self.sample_embedder.embeddings(self.distill_NLP_syntax_marking_ids[:nlp_reserved_len].unsqueeze(-1).expand([nlp_reserved_len, _i]))
            description_embs_tie = self.nlp_glue_layer(description_embs_tie)
            description_embs_tie += syntax_markings_tie
            tensor_tie = torch.cat([description_embs_tie, tensor_tie], dim=0)
            token_lens_per_inst += len(self.distill_NLP_syntax_marking_ids)
            tensor_ite = tensor_tie.transpose(0,1)
            
        return tensor_ite, token_lens_per_inst



    def get_length_after_batching(self, seqs: List[Sequence]) -> torch.Tensor:
        lengths = torch.zeros(len(seqs), dtype=torch.long)
        for i, seq in enumerate(seqs):
            lengths[i] = len(seq)
        assert lengths.max() <= self.max_seq_len, "issue with lengths after batching"
        return lengths


def getMLP(neurons, activation=nn.GELU, bias=True, dropout=0.2, last_dropout=False, normfun='layernorm'):
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    if len(neurons) in [0,1]:
        return nn.Identity()
        
    nn_list = []
    n = len(neurons)-1
    for i in range(n-1):
        if normfun=='layernorm':
            norm = nn.LayerNorm(neurons[i+1])
        elif normfun=='batchnorm':
            norm = nn.BatchNorm1d(neurons[i+1])
        else:
            norm = nn.Identity()
        nn_list.extend([nn.Linear(neurons[i], neurons[i+1], bias=bias), norm, activation(), nn.Dropout(dropout)])
    
    nn_list.extend([nn.Linear(neurons[n-1], neurons[n], bias=bias)])
    if last_dropout:
        if normfun=='layernorm':
            norm = nn.LayerNorm(neurons[-1])
        elif normfun=='batchnorm':
            norm = nn.BatchNorm1d(neurons[-1])
        else:
            norm = nn.Identity()
        nn_list.extend([norm, activation(), nn.Dropout(dropout)])

    return nn.Sequential(*nn_list)



def get_model_tokenizer(which_model, device='cpu'):
    if which_model=='bert':
        from transformers import BertTokenizer, BertModel
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        _model = BertModel.from_pretrained("bert-large-uncased")
        pad_token_id = tokenizer.encode(tokenizer.pad_token)[1]
        

    elif which_model=='distilbert':
        from transformers import DistilBertTokenizer, DistilBertModel
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        _model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        pad_token_id = tokenizer.encode(tokenizer.pad_token)[1]


    _model = _model.to(device)


    def model(inp_tensor, attn_mask):


        out_BTE = _model(input_ids = inp_tensor, attention_mask = attn_mask).last_hidden_state
        first_pool = out_BTE[:,0]       # BE
        max_pool = out_BTE.max(dim=1).values
        min_pool = out_BTE.min(dim=1).values
        last_pool = out_BTE[:,0]

        description_embs_tbe = torch.stack([first_pool, last_pool, max_pool, min_pool], dim=0)
        return description_embs_tbe


    return model, tokenizer, pad_token_id

