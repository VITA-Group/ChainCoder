import os
from collections import defaultdict
import itertools
import os
import pickle
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, DistilBertTokenizer
import time
import sys
from argparse import ArgumentParser
import copy
from glob import glob

from tokenizer.tokenizerAPI import (
    tokenizerAPI_IN2R,
    tokenizerAPI_IR2N,
    tokenizerAPI_ON2R,
    tokenizerAPI_OR2N,
    tokenizerAPI_OT2R,
    tokenizerAPI_OR2T,
    vocabulary_defs,
)

from dataloaders.check_exec_match import check_io_match_one_sample_obj
from dataloaders.data_augmentation import iodata_augmentor
from dataloaders.loader_utils import evalio, load_all_instances, shuffled, save_raw, parse_loadername_from_filename, MyTimeoutError, timeout





vocabulary_defs.refuse_unseen_tokens = True

def parse_args():

    parser = ArgumentParser()

    one_sample_program_run_timelimit = 10


    parser.add_argument(
        "raw_data_dir", 
        help='This dir is read-only for this script: it might read raw files from this dir (or, might read from --code_augmented_dir, depending on how you set --SG_from), then convert to int and save to --pickle_dir.'
    )
    parser.add_argument(
        "reg_dir", 
        help='output regularized result dir'
    )

    
    parser.add_argument("--verbose", type=int, default=1)


    # 🟩 Three important dirs below.
    parser.add_argument(
        "--SG_from", 
        default='raw',
        choices=['raw', 'code_augmented'],
        help='Choose where to read raw file and convert to int; choices are ["raw", "code_augmented"].'
    )


    parser.add_argument(
        "--code_augmented_dir", 
        default='/path/to/your/code_augmented_dir',
        help='This dir is read-only for this script: it might read raw files from this dir (or, might read from --raw_data_dir, depending on how you set --SG_from), then convert to int and save to --pickle_dir.'
    )


    parser.add_argument(
        "--one_sample_program_run_timelimit", 
        type=int, 
        default=one_sample_program_run_timelimit, 
    )
    
    parser.add_argument(
        "--only_do_subfolders", 
        type=str, 
        # default='0,1,2,3', 
        default='all', 
        help='Used to slice subfolder; values 0~3, seperate by comma, or "all".'
    )
    parser.add_argument(
        "--only_do_ires", 
        type=str, 
        # default='?0,?1,?2,?3,?4,?5,?6,?7,?8,?9', 
        default='all', 
        help='Used to slice progress; usage: --only_do_ires="??", where "?" can be 0~9, seperate by comma; or, --only_do_ires="all".'
    )


    args = parser.parse_args()
    if args.SG_from=='code_augmented':
        os.makedirs(args.code_augmented_dir, exist_ok=True)



    if args.only_do_subfolders=='all':
        args.only_do_subfolders = list(range(4))
    else:
        args.only_do_subfolders = [int(x) for x in args.only_do_subfolders.split(',')]
    if args.only_do_ires=='all':
        args.only_do_ires = [f'?????{x}' for x in range(10)]
    else:
        tmp = []
        for x in args.only_do_ires.split(','):
            x = '?'*(6-len(x)) + x
            tmp.append(x)
        args.only_do_ires = tmp

    if args.verbose:
        print('🙂', file=open('_log_ioaug_err.py', 'w'))


    return args


args = parse_args()



def check_match_loop(code_raw_st, io_s2t_orig):
    io2codes = defaultdict(list)
    totalnum = len(io_s2t_orig)
    print(f'🟧 Num samples = {totalnum}')
    for i_code in tqdm(range(len(code_raw_st))):
        code = code_raw_st[i_code]
        io_s2t = []
        core_exec_time = 0
        validnum = 0
        for i, ioobj in enumerate(io_s2t_orig):
            is_match, exec_out, prt_str = check_io_match_one_sample_obj(ioobj, code, sanity_check_timeout=args.one_sample_program_run_timelimit)

            _ct = float(prt_str.split('core_exec_time:\n\t ')[1])
            if _ct!=-1: # only add those passed time.
                core_exec_time += _ct


            io_s2t.append(copy.deepcopy(ioobj))
            # y = process(x, code)

            if is_match:
                validnum += 1
            else:
                if not (type(exec_out) is RuntimeError):
                    io_s2t[-1][1] = exec_out
                    validnum += 1
                else:
                    io_s2t.pop()

        io2codes[repr(io_s2t)].append([code, core_exec_time, validnum, totalnum])

    return io2codes


finished_f = 'finished_reg_run.txt'
failed_f = 'failed_reg_run.txt'

def main():
    try:
        main_sub()
    except:
        print(('failed somewhere', args.only_do_subfolders, args.only_do_ires), file=open(failed_f, 'a'))
    return


def main_sub():
    if args.SG_from=='raw':
        args.SG_root_dir = args.raw_data_dir
    elif args.SG_from=='code_augmented':
        args.SG_root_dir = args.code_augmented_dir

    subfolders = [
        'difficulty_introductory',
        'difficulty_interview',
        'difficulty_competition',
        'difficulty_dm_code_contest',
        ]

    subfolders = [subfolders[i] for i in args.only_do_subfolders]


    converted_files = 0
    total_faith_div_cnts = np.array([0,0,0])

    for subfolder in subfolders:
        SG_subdir = os.path.join(args.SG_root_dir, subfolder)

        reg_dir_sub = os.path.join(args.reg_dir, subfolder)
        os.makedirs(reg_dir_sub, exist_ok=True)
        print(f'🙇 Ready to regularize to {reg_dir_sub}! 🙇')

        for ire in tqdm(args.only_do_ires):
            file_id_re = ire

            all_instances = load_all_instances(SG_subdir, file_id_re, shuffle=False)
            if len(all_instances[0])==0:
                continue


            allinst_cvt = list(zip(*all_instances))


            for i_inst, (code_raw_st, code_nameRep_st, x_raw_st, y_raw_st, io_s2t_orig, description, filename) in enumerate(allinst_cvt):

                # 🟩 From here do whatever with these variables: they loop for the entire dataset per instance
                
                print(f'🟧 beginning check match: \n\t ire/subfolder = {ire, subfolder} \n\t inst/all ins = {i_inst} / {len(allinst_cvt)}\n\t code num = {len(code_raw_st)}')

                io2codes = check_match_loop(code_raw_st, io_s2t_orig)

                which_loader, inst_id_orig = parse_loadername_from_filename(filename)
                
                for io_r, codes in io2codes.items():
                    total_faith_div_cnts[0] += len(codes)

                    io_objs = evalio(io_r)
                    if io_objs==io_s2t_orig:
                        pdescription = description
                        _hash_code_behavior = 'orig'
                        total_faith_div_cnts[1] += len(codes)
                    else:
                        pdescription = ''
                        _hash_code_behavior = hash(repr(io_objs))
                        total_faith_div_cnts[2] += len(codes)
    
                    codes_nameReplaced, ctimes, codes_raw = [], [], []
                    for code, ctime, validnum, totalnum in codes:
                        try:
                            namer = tokenizerAPI_OT2R(*tokenizerAPI_OR2T(code))
                        except:
                            print(f'# ❓❓ an impossible error occured: tokenization error: \n\t\t# {sys.exc_info()[:-1]}\n# Code is:\n{code}', file=open('_log_for_reg.py', 'a'))
                            continue
                        codes_nameReplaced.append(namer)
                        codes_raw.append(code)
                        ctimes.append([ctime, validnum, totalnum])
                        
                    
                    raw_readable_with_time = list(map(lambda ab: f'{ab[0]}\n# ⏳ ⏳ Meta Info\n\t# time = {ab[1]}\n\t# io samples valid / all = {ab[2]} / {ab[3]}\n', codes))

                    instance_id = f'{inst_id_orig}ire{hash(ire)}hash{_hash_code_behavior}'
                    cross_samp_join = '\n\n# 🟨 🟨 🟨 🟨 \n\n'
                    codes_readable_raw = cross_samp_join.join(raw_readable_with_time)
                    codes_readable_nameReplaced = cross_samp_join.join(codes_nameReplaced)
                    iodatas_readable = cross_samp_join.join([repr(tuple(x)) for x in io_objs])

                    if validnum!=0:
                        save_dir = reg_dir_sub
                    else:
                        save_dir = reg_dir_sub + '_bad_codes'  # bad_codes means fail to finish all inputs.

                    save_raw(save_dir, which_loader, instance_id, 
                        codes_raw, codes_nameReplaced, codes_readable_raw, codes_readable_nameReplaced,
                        ctimes, [], io_objs, iodatas_readable, 
                        pdescription)


                print(f'🟧 👌 check match finish: \n\t subfolder = {subfolder} \n\t prog = (ire): {ire} / {args.only_do_ires}  (inst): {i_inst} / {len(allinst_cvt)}  \n\t total_faith_div_cnts = {total_faith_div_cnts}\n\t total_faith_div_cnts = {total_faith_div_cnts/10000} ')

        print((subfolder, file_id_re), file=open(finished_f, 'a'))

    print('🤘 All finish! 🤘')

    return



def print_stats(iodata_is, code_is):
    num_inst = len(iodata_is)
    samples_1 = [len(x) for x in iodata_is]
    samples_2 = [len(x) for x in code_is]
    mv1 = [np.median(samples_1), np.std(samples_1)]
    mv2 = [np.median(samples_2), np.std(samples_2)]

    flatten1 = itertools.chain.from_iterable(iodata_is)
    lens1 = list(map(lambda x: len(x), flatten1))
    lens1 = [np.median(lens1), np.std(lens1)]
    flatten2 = itertools.chain.from_iterable(code_is)
    lens2 = list(map(lambda x: len(x), flatten2))
    lens2 = [np.median(lens2), np.std(lens2)]

    print(f'Sample Num Stats of {num_inst} I/O data insts:\t\tNum Samples = {mv1[0]} ± {mv1[1]:.3f}\t\tToken Len = {lens1[0]} ± {lens1[1]:.3f}')
    print(f'Sample Num Stats of {num_inst} code data insts:\t\tNum Samples = {mv2[0]} ± {mv2[1]:.3f}\t\tToken Len = {lens2[0]} ± {lens2[1]:.3f}')
    return



if __name__ == "__main__":

    main()

