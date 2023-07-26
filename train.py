import numpy as np
import torch
import os
from tqdm import tqdm

from trainer.slurm import init_distributed_mode
from model import build_modules
from trainer.trainer import Trainer
from parsers import get_parser
from tokenizer.tokenizerAPI import vocabulary_defs, closest
from dataloaders.sttd import get_ChainCoder_dataloader, pretrain_loader


def main():

    params = get_parser()

    init_distributed_mode(params)
    # CPU / CUDA
    if not params.run_on_cpu:
        assert torch.cuda.is_available()

    os.makedirs(params.training_ckpt_dump_path, exist_ok=1)
    modules = build_modules(vocabulary_defs, params)
    trnr = Trainer(modules, vocabulary_defs, params)

    if params.training_resume_ckpt_from != '':
        trnr.reload_checkpoint(params.training_resume_ckpt_from)


    # ---- training
    for iepoch in range(params.max_epoch):

        def control_difficulty(params, iepoch):
            params.batch_size

            grid_io = [1] + list(range(4, params.training_difficulty_A_io, 4))     # [1, 4, 8, 12, ...]
            grid_code = list(range(1,params.training_difficulty_A_code))
            def periodic(i, A, T):
                def period_decimal(x):
                    decimal = x-np.floor(x)
                    return min(decimal, 1-decimal)
                return 2* A * period_decimal(i/T+0.5)

            mixed_io = lambda i: closest(periodic(i, params.training_difficulty_A_io, params.training_difficulty_T_io), grid_io)  # range(20) -> [28, 24, 16, 8, 4, 12, 20, 28, 28, 20, 8, 1, 8, 16, ...]
            mixed_code = lambda i: closest(periodic(i, params.training_difficulty_A_code, params.training_difficulty_T_code), grid_code)  # range(20) -> [7, 7, 6, 5, 5, 4, 3, 2, 1, 1, 1, 1, 2, 3, ...]
            

            params.samples_per_instance_io = mixed_io(iepoch)
            params.samples_per_instance_code = mixed_code(iepoch)
            return

        control_difficulty(params, iepoch)

        if params.is_pretraining:
            trainloader = pretrain_loader(params)
        else:
            trainloader = get_ChainCoder_dataloader(params, params.pickle_data_root)

        print(f'\n🟨 🟨 🟨 🟨    Training Epoch {iepoch} Start    🟨 🟨 🟨 🟨\n')

        for samples in tqdm(trainloader):
            
            if samples is None:  # for debug purpose, or robust_tokenization_after_fix_vocab, output may be None
                continue
            trnr.enc_dec_step(samples)

        if iepoch%20==0:
            trnr.save_checkpoint(f'epoch-{iepoch}.pth')


if __name__ == "__main__":

    main()
