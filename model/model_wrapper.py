import numpy as np
import torch
import torch.nn as nn

from tokenizer.tokenizerAPI import tokenizerAPI_ON2R_flatten

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

class ModelWrapper(nn.Module):
    """""" 
    def __init__(self,
                env=None,
                trnr=None,
                embedder=None,
                encoder=None,
                decoder=None,
                beam_type="search",
                beam_length_penalty=1,
                beam_size=1,
                beam_early_stopping=True,
                max_generated_output_len=200,
                beam_temperature=1.,
                ):
        super().__init__()

        self.env = env
        self.trnr = trnr
        self.embedder = embedder
        self.encoder = encoder
        self.decoder = decoder
        self.beam_type = beam_type
        self.beam_early_stopping = beam_early_stopping
        self.max_generated_output_len = max_generated_output_len
        self.beam_size = beam_size
        self.beam_length_penalty = beam_length_penalty
        self.beam_temperature = beam_temperature
        self.device=next(self.embedder.parameters()).device

    def set_args(self, args={}):
        for arg, val in args.items():
            assert hasattr(self, arg), "{} arg does not exist".format(arg)
            setattr(self, arg, val)
    def decode_from_batch_output(self, generations_iat, is_wrapped_by_hyp=0):
        bs = len(generations_iat)
        if is_wrapped_by_hyp:
            generations_iat = [list(filter(lambda x: x is not None, [self.env.idx_to_infix(hyp.cpu().tolist()[1:], is_float=False, str_array=False) for (_, hyp) in generations_iat[i]])) for i in range(bs)]  # [3(instance), 4(beam_size), tokens]
        else:
            generations_iat = [list(filter(lambda x: x is not None, [tokenizerAPI_ON2R_flatten(hyp[1:-1]) for hyp in generations_iat[i]])) for i in  range(bs)]                             # [3,1,body] = [instance, answer, token]
        return generations_iat

    @torch.no_grad()
    def forward(
        self,
        samples
    ):

        """
        x: bags of sequences (B, T)
        """

        embedder, encoder, decoder = self.embedder, self.encoder, self.decoder

        # x1_TBE, x_len = embedder(samples)
        # encoded = encoder("fwd", x=x1_TBE, lengths=x_len, causal=False).transpose(0,1)

        encoded_BTE, x_len = self.trnr.io_embed_encode(samples)

        # encoded = encoded_BTE.transpose(0,1)
        encoded = encoded_BTE

        # x_len = torch.tensor([encoded_BTE.shape[0]]).to(encoded_BTE.device)

        outputs = []

        bs = encoded.shape[0]

        ### Greedy solution.
        generations, gen_len = decoder.generate(      # generations: torch.Size([68, 3]) = TB; gen_len = [B]
                    encoded,                          # BTE
                    x_len,                            # [B]
                    sample_temperature=None,
                    max_len=self.max_generated_output_len,
                    )

        generations = generations.unsqueeze(-1).view(generations.shape[0], bs, 1)       # torch.Size([68, 3, 1])
        generations = generations.transpose(0,1).transpose(1,2).cpu().tolist()          # (3, 1, 68)


        generations = self.decode_from_batch_output(generations, is_wrapped_by_hyp=0)

        if self.beam_type == "search":
            decoded, tgt_len, search_generations = decoder.generate_beam( # decoded: torch [68, 3]; tgt_len: [3]; search_generations: [batch hyp, beam_num, score and tokens]
                encoded,
                x_len,
                beam_size=self.beam_size,
                length_penalty=self.beam_length_penalty,
                max_len=self.max_generated_output_len,
                early_stopping=self.beam_early_stopping,
            )
            search_generations = [sorted([hyp for hyp in search_generations[i].hyp], key=lambda s: s[0], reverse=True) for i in range(bs)]  # remove the hyp wrapper
            
            search_generations = self.decode_from_batch_output(search_generations, is_wrapped_by_hyp=1)

            for i in range(bs):
                generations[i].extend(search_generations[i])  # generations and search_generations both: IAT

        elif self.beam_type == "sampling":
            num_samples = self.beam_size
            encoded = (encoded.unsqueeze(1)
                .expand((bs, num_samples) + encoded.shape[1:])
                .contiguous()
                .view((bs * num_samples,) + encoded.shape[1:])
            )
            x_len = x_len.unsqueeze(1).expand(bs, num_samples).contiguous().view(-1)
            sampling_generations, _ = decoder.generate(
                encoded,
                x_len,
                sample_temperature = self.beam_temperature,
                max_len=self.max_generated_output_len
                )
            sampling_generations = sampling_generations.unsqueeze(-1).view(sampling_generations.shape[0], bs, num_samples)
            sampling_generations = sampling_generations.transpose(0, 1).transpose(1, 2).cpu().tolist()
            
            
            sampling_generations = self.decode_from_batch_output(sampling_generations, is_wrapped_by_hyp=0)
            
            for i in range(bs):
                generations[i].extend(sampling_generations[i])
        else: 
            raise NotImplementedError


        outputs.extend(generations)
        return outputs  # shape= IAT


