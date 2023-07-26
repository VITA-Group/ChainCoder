from logging import getLogger
import os
import torch


from .embedders import ChainCoderSampleEmbedder

from .transformer import TransformerModel



logger = getLogger()


def check_model_params(params):
    """
    Check models parameters.
    """
    # model dimensions
    assert params.enc_emb_dim % params.n_enc_heads == 0
    assert params.dec_emb_dim % params.n_dec_heads == 0

    # reload a pretrained model
    if params.reload_model != "":
        print("Reloading model from ", params.reload_model)
        assert os.path.isfile(params.reload_model)


def build_modules(vocabulary_defs, params):
    """
    Build modules.
    """
    modules = {}
    
    modules["embedder"] = ChainCoderSampleEmbedder(params, vocabulary_defs)
    # vocabulary_defs.get_length_after_batching = modules["embedder"].get_length_after_batching

    modules["encoder"] = TransformerModel(
        params,
        vocabulary_defs,
        is_encoder=True,
        with_output=False,
        use_prior_embeddings=True,
        positional_embeddings=params.enc_positional_embeddings
    )




    modules["decoder"] = TransformerModel(
            params,
            vocabulary_defs,
            is_encoder=False,
            with_output=True,
            use_prior_embeddings=False,
            positional_embeddings=params.dec_positional_embeddings
    )

    # log
    for k, v in modules.items():
        logger.debug(f"{v}: {v}")
    for k, v in modules.items():
        logger.info(
            f"Number of parameters ({k}): {sum([p.numel() for p in v.parameters() if p.requires_grad])}"
        )

    # cuda
    if not params.run_on_cpu:
        for v in modules.values():
            v.cuda()


    if params.torch_parallel:
        modules["embedder"] = torch.nn.DataParallel(modules["embedder"])
        modules["encoder"] = torch.nn.DataParallel(modules["encoder"])
        modules["decoder"] = torch.nn.DataParallel(modules["decoder"])

    return modules



def load_modules(reload_file, modules):
    # reload pretrained modules
    reloaded = torch.load(reload_file)
    for k, v in modules.items():
        assert k in reloaded
        if all([k2.startswith("module.") for k2 in reloaded[k].keys()]):
            reloaded[k] = {
                k2[len("module.") :]: v2 for k2, v2 in reloaded[k].items()
            }
        v.load_state_dict(reloaded[k])
    print(f"\n\n Reloading modules from {reload_file} SUCCEED! \n")
