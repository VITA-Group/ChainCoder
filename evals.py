from collections import defaultdict
import os
import torch
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt




from parsers import get_parser

from trainer.slurm import init_signal_handler, init_distributed_mode
from model import check_model_params, build_modules, load_modules
from model.model_wrapper import ModelWrapper
from model.embedders import get_model_tokenizer
from trainer.trainer import Trainer


from dataloaders.loader_utils import timeout
from dataloaders.sttd import get_ChainCoder_dataloader
from dataloaders.check_exec_match import check_io_match_one_sample_int, check_io_match_one_sample_obj
from tokenizer.tokenizerAPI import (
    vocabulary_defs, load_txt,
    tokenizerAPI_IN2R, 
    tokenizerAPI_ON2R, 
    tokenizerAPI_OR2T,
)


def run_evals(params):

    params.multi_gpu=False
    params.is_slurm_job = False
    params.local_rank = -1
    params.master_port = -1
    params.num_workers = 1
    params.target_noise=0.0
    params.max_input_points=200
    os.environ['CUDA_VISIBLE_DEVICES'] = params.CUDA_VISIBLE_DEVICES

    init_distributed_mode(params)
    if params.is_slurm_job:
        init_signal_handler()

    # CPU / CUDA
    if not params.run_on_cpu:
        assert torch.cuda.is_available()
    params.eval_only=True

    # build environment / modules
    if params.batch_size_eval is None:
        params.batch_size_eval = int(1.5 * params.batch_size)

    env = vocabulary_defs
    modules = build_modules(env, params)
    load_modules(params.testing_load_ckpt_from, modules)####
    trnr = Trainer(modules, vocabulary_defs, params)


    embedder = (
        modules["embedder"].module
        if params.multi_gpu
        else modules["embedder"]
    )

    encoder = (
        modules["encoder"].module
        if params.multi_gpu
        else modules["encoder"]
    )
    decoder = (
        modules["decoder"].module
        if params.multi_gpu
        else modules["decoder"]
    )
    embedder.eval()
    encoder.eval()
    decoder.eval()


    model = ModelWrapper(
                    env=env, 
                    trnr=trnr,
                    embedder=embedder, 
                    encoder=encoder, 
                    decoder=decoder,
                    beam_length_penalty=params.beam_length_penalty,
                    beam_size=params.beam_size,
                    max_generated_output_len=params.max_generated_output_len,
                    beam_early_stopping=params.beam_early_stopping,
                    beam_temperature=params.beam_temperature,
                    beam_type=params.beam_type,
                    )
    if not params.run_on_cpu:
        model = model.to('cuda')


    def control_evaluator(samples_per_instance_io):
        params.samples_per_instance_io = samples_per_instance_io
        params.samples_per_instance_io_hold = 4
        params.batch_size = 1  # at test time, always use batch-size = 1
        params.samples_per_instance_code = 2
        params.fine_fune_nlp = 0
        params.beam_size = 4
        return


    given_samples_list = [4]
    for i in tqdm(range(len(given_samples_list))):
        control_evaluator(given_samples_list[i])
        testloader = get_ChainCoder_dataloader(params, params.test_pickle_dir)

        print(f'\n\n  num samples feed is:     {given_samples_list[i]} \n ')

        acc_syntax_error_free, acc_error_free, acc_demo_pass, acc_all_pass = evaluate_syntax_transformer(testloader, model, params)

    return



def evaluate_syntax_transformer(testloader, model, params):
    acc_syntax_error_free = []

    for i, samples in enumerate(tqdm(testloader)):
        if samples==None:
            continue

        programs_ia = model(samples)  # 2D list, dims = [instance, answers] (num of instance always == 1 in test phase); output None means syntax error/etc so as to fail parsing code.
        
        assert len(programs_ia)==1
        answers = programs_ia[0]

        io_objs = [tokenizerAPI_IN2R(samples['ioData_ist2'][0][ioSamp_id]) for ioSamp_id in range(len(samples['ioData_ist2'][0]))]
        io_objs = list(map(lambda x: tuple(x), io_objs))

        if len(answers)!=0:
            acc_syntax_error_free.append(1)
        else:
            acc_syntax_error_free.append(0)

        is_match = False
        is_all_bug_free = False
        for answer in answers:
            ioSamp_id = 0
            io_ns = samples['ioData_ist2'][0][ioSamp_id]
            io_obj = tokenizerAPI_IN2R(io_ns)
            
            is_match, exec_out, prt_str = check_io_match_one_sample_obj(io_obj, answer, params.program_forward_run_timeout)
            if type(exec_out) is not RuntimeError:
                is_all_bug_free = True
            if is_match:
                is_all_bug_free = True
                break
            else:
                print(prt_str)



if __name__ == "__main__":

    params = get_parser()
    run_evals(params)
