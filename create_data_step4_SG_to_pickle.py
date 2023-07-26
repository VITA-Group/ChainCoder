import itertools
import os
import pickle
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, DistilBertTokenizer
import time
import sys
from argparse import ArgumentParser



from tokenizer.tokenizerAPI import (
    tokenizerAPI_IN2R,
    tokenizerAPI_IR2N,
    tokenizerAPI_ON2R,
    tokenizerAPI_OR2N,
    vocabulary_defs,
)


from dataloaders.check_exec_match import check_io_match_one_sample_obj
from dataloaders.data_augmentation import iodata_augmentor
from dataloaders.loader_utils import load_all_instances, shuffled, timeout


vocabulary_defs.refuse_unseen_tokens = True

def parse_args():

    parser = ArgumentParser()


    one_instance_io_aug_timelimit = 1000


    do_io_augment_now = 1
    if do_io_augment_now:
        one_instance_timelimit = 400
    else:
        one_instance_timelimit = 20
    one_instance_timelimit = 999999


    one_sample_program_run_timelimit = 10





    instances_per_file = 8
    instances_per_file = 64
    machine_name = 'some_machine'
    SG_from = ['raw', 'code_augmented'][0]
    io_augmented_samples_per_instance = 5

    parser.add_argument(
        "pickle_dir", 
        help='This dir is where the int converted files are dumped into. If it does not exist, will be created automatically.'
    )
    parser.add_argument(
        "raw_data_dir", 
        help='This dir is read-only for this script: it might read raw files from this dir (or, might read from --code_augmented_dir, depending on how you set --SG_from), then convert to int and save to --pickle_dir.'
    )


    parser.add_argument("--machine_name", default=machine_name)
    parser.add_argument("--do_io_augment_now", type=int, default=do_io_augment_now)
    parser.add_argument("--check_match_now", type=int, default=-1, help='Recommnded set to -1 to disable check I/O match during this stage.')


    parser.add_argument("--instances_per_file", default=instances_per_file, type=int, help='This number should better be big, must >= batch_size during step2_API.')
    parser.add_argument("--io_augmented_samples_per_instance", default=io_augmented_samples_per_instance, type=int)
    parser.add_argument("--verbose", type=int, default=1)


    # 🟩 Three important dirs below.
    parser.add_argument(
        "--SG_from", 
        default=SG_from,
        choices=['raw', 'code_augmented'],
        help='Choose where to read raw file and convert to int; choices are ["raw", "code_augmented"].'
    )

    parser.add_argument(
        "--code_augmented_dir", 
        default='/path/to/your/iodata/augmentation/dir',
        help='This dir is read-only for this script: it might read raw files from this dir (or, might read from --raw_data_dir, depending on how you set --SG_from), then convert to int and save to --pickle_dir.'
    )


    # 🟩 Four time limit controls below.
    parser.add_argument(
        "--one_sample_tokenize_encode_timelimit", 
        type=int, 
        default=2, 
        help='Timelimit for python repr -> token -> int procedure. Recommended 2s.'
    )
    parser.add_argument(
        "--one_sample_program_run_timelimit", 
        type=int, 
        default=one_sample_program_run_timelimit, 
    )
    parser.add_argument(
        "--one_instance_io_aug_timelimit", 
        type=int, 
        default=one_instance_io_aug_timelimit, 
        help='Better set this to longer time, otherwise this instance will be discarded if did not finish. To make sure it indeed does not fail, you can set this number to some large interger times of "one_sample_program_run_timelimit". -1 means unlimited time.'
    )
    parser.add_argument(
        "--one_instance_timelimit", 
        type=int, 
        default=one_instance_timelimit, 
        help='Time limit for one instance "process". This "process" at least contain the int conversion step. Or, if  --do_io_augment_now is set to True,  can further contain the I/O augmentation step. -1 means unlimited time.'
    )


    args = parser.parse_args()
    os.makedirs(args.code_augmented_dir, exist_ok=True)
    os.makedirs(args.pickle_dir, exist_ok=True)


    if args.one_instance_io_aug_timelimit<0:
        args.one_instance_io_aug_timelimit = 999999999
    if args.one_instance_timelimit<0:
        args.one_instance_timelimit = 999999999
    if args.verbose:
        print('🙂', file=open('_log_ioaug_err.py', 'w'))



    assert len(os.listdir(args.raw_data_dir))==4 and 'difficulty_introductory' in os.listdir(args.raw_data_dir)

    return args


args = parse_args()


@timeout(args.one_instance_io_aug_timelimit)
def iodata_augmentor_timed_wrap(code_raw_st, io_s2t_orig):
    if args.verbose:
        print('\n🟥 begin iodata aug...\n')
    iAug_sat, oAug_sat = iodata_augmentor(code_raw_st, io_s2t_orig, args.io_augmented_samples_per_instance, args.one_sample_program_run_timelimit, args)
    
    # io_s2t = list(zip(iAug_sat, oAug_sat))
    ioAug_s2t = list(zip(iAug_sat, oAug_sat))
    if args.verbose:
        print(f'\n🤘 🤘 iodata aug finish, generated samples = {len(iAug_sat)}\n')


    return ioAug_s2t


def check_match_loop(code_raw_st, io_s2t_orig, cnts):
    valid_ids = []
    for i_code in tqdm(range(len(code_raw_st))):
        is_match = True
        code = code_raw_st[i_code]
        for i, ioobj in enumerate(io_s2t_orig):
        # for i, x in enumerate(idatas_to_check):
            is_match, exec_out, prt_str = check_io_match_one_sample_obj(ioobj, code, sanity_check_timeout=10)
            # y = process(x, code)
            # ys.append(y)
            # is_match = (orig_odata_s2t[i]==y)
            if not is_match:
                cnts[0] += 1
                prt_str = f'orig code {i_code} / {len(code_raw_st)} failed the test/n{prt_str}'
                print(prt_str)
                break
        if is_match:
            cnts[1] += 1
        valid_ids.append(i)
    return cnts, valid_ids

def main():
    do_io_augment_now = args.do_io_augment_now

    if args.SG_from=='raw':
        args.SG_root_dir = args.raw_data_dir
    elif args.SG_from=='code_augmented':
        args.SG_root_dir = args.code_augmented_dir

    tokenizer_bert = BertTokenizer.from_pretrained('bert-large-uncased')
    tokenizer_distilbert = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    subfolders = shuffled([
        'difficulty_introductory',
        'difficulty_interview',
        'difficulty_competition',
        'difficulty_dm_code_contest',
        ])

    converted_files = 0
    fail_pass_cnts = [0,0]
    for subfolder in subfolders:
        SG_subdir = os.path.join(args.SG_root_dir, subfolder)

        if args.do_io_augment_now:
            dump_pickle_dir = os.path.join(args.pickle_dir, f'{subfolder}_ioAug')
        else:
            dump_pickle_dir = os.path.join(args.pickle_dir, subfolder)

        os.makedirs(dump_pickle_dir, exist_ok=1)
        print(f'🙇 Ready to generate to {dump_pickle_dir}! 🙇')

        valid_iodatas_int_is = []
        valid_codes_int_is = []
        valid_desc_bert = []
        valid_desc_distilbert = []

        # metadata = {}
        for ire in tqdm(shuffled(range(10))):
            file_id_re = f'?{ire}????'
            file_id_re = f'?????{ire}'
            all_instances = load_all_instances(SG_subdir, file_id_re)
            if len(all_instances[0])==0:
                continue


            allinst_cvt = list(zip(*all_instances))
            for i_inst, (code_raw_st, code_nameRep_st, x_raw_st, y_raw_st, io_s2t_orig, description, filename) in enumerate(shuffled(allinst_cvt)):

                # 🟩 From here do whatever with these variables: they loop for the entire dataset per instance
                check_match_loop_all=1
                if check_match_loop_all:
                    print(f'🟧 beginning check match: \n\t prog = {i_inst} / {len(allinst_cvt)}\n\t code numbers = {len(code_raw_st)}\n\t ire/subfolder = {ire, subfolder}')
                    fail_pass_cnts = check_match_loop(code_raw_st, io_s2t_orig, fail_pass_cnts)
                    print(f'🟧 🟧 check match: \n\t prog = {i_inst} / {len(allinst_cvt)}\n\t fail-pass-cnt = {fail_pass_cnts, np.array(fail_pass_cnts)/10000}\n\t subfolder = {subfolder}')
                    continue


                try:

                    @timeout(args.one_instance_timelimit)
                    def generation_step_wrap():
                        # nonlocal io_s2t, code_nameRep_st

                        # 🟩 encode iodata
                        instance_io = []
                        instance_code = []
                        if do_io_augment_now:
                            io_s2t = iodata_augmentor_timed_wrap(code_raw_st, io_s2t_orig)
                        else:
                            io_s2t = io_s2t_orig

                        for iodata in io_s2t:
                            try:
                                @timeout(args.one_sample_tokenize_encode_timelimit)
                                def time_wrap_io():
                                    return tokenizerAPI_IR2N(iodata)
                                io_ns = time_wrap_io()
                            except:
                                if args.verbose:
                                    print("In datagen step3, tok->int for io, error:", sys.exc_info()[:-1])
                                    print("In datagen step3, tok->int for io, error:", sys.exc_info()[:-1], file=open('_log_ioaug_err.py', 'a'))
                                continue
                            instance_io.append(io_ns)


                        # 🟩 encode codes

                        for code in code_nameRep_st:
                            try:
                                @timeout(args.one_sample_tokenize_encode_timelimit)
                                def time_wrap_c():
                                    return tokenizerAPI_OR2N(code)
                                code_ns = time_wrap_c()
                            except:
                                if args.verbose:
                                    print("In datagen step3, tok->int for code, error:", sys.exc_info()[:-1])
                                    print("In datagen step3, tok->int for code, error:", sys.exc_info()[:-1], file=open('_log_ioaug_err.py', 'a'))
                                continue
                            instance_code.append(code_ns)

                        # 🟩 drop thin instance 
                        if len(instance_code)==0 or len(instance_io)<=1:
                            print('in generating pickle, valid sample num too small, discarded this INSTANCE')
                            return valid_iodatas_int_is, valid_codes_int_is, valid_desc_distilbert, valid_desc_bert
                        else:
                            # 🟩 encode descriptions
                            distilbert_ids = tokenizer_distilbert(description)
                            bert_ids = tokenizer_bert(description)

                            # 🟩 Finish this instance.
                            valid_iodatas_int_is.append(instance_io)
                            valid_codes_int_is.append(instance_code)
                            valid_desc_distilbert.append(distilbert_ids)
                            valid_desc_bert.append(bert_ids)


                            # 🟩 check_match_now

                        return valid_iodatas_int_is, valid_codes_int_is, valid_desc_distilbert, valid_desc_bert

                    valid_iodatas_int_is, valid_codes_int_is, valid_desc_distilbert, valid_desc_bert = generation_step_wrap()
                except:
                # else:
                    if args.verbose:
                        print(f'instance int convert err: {sys.exc_info()[:-1]}. This ENTIRE INSTANCE is discarded.')
                        print(f'instance int convert err: {sys.exc_info()[:-1]}. This ENTIRE INSTANCE is discarded.', file=open('_log_ioaug_err.py', 'a'))
                    continue



                # 🟩 check if need to dump into pkl
                if len(valid_codes_int_is)>=args.instances_per_file:
                    converted_sub = [valid_iodatas_int_is, valid_codes_int_is, valid_desc_distilbert, valid_desc_bert]
                    pkl_file = save_one_pkl(dump_pickle_dir, converted_sub, args)
                    print_stats(valid_iodatas_int_is, valid_codes_int_is)
                    print(f'pickle saved to {pkl_file}\n')
                    converted_files += 1

                    valid_codes_int_is = []
                    valid_iodatas_int_is = []
                    valid_desc_distilbert = []
                    valid_desc_bert = []
                    vocabulary_defs.show_vocab()

                if do_io_augment_now and args.verbose:
                    msg = f'🟥 🟥 another instance finished, for 🟥 io aug 🟥  Now subfolder = {subfolder} ; now total converted instance = {converted_files} files * {args.instances_per_file} insts_per_file = {converted_files*args.instances_per_file} instances.👌'
                    print(msg)
                    print(msg, file=open('_log_ioaug_err.py', 'a'))


        print(f'Just finished subfolder {subfolder} ; now total converted instance = {converted_files} files * {args.instances_per_file} insts_per_file = {converted_files*args.instances_per_file} instances.👌')

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


def save_one_pkl(dump_pickle_dir, converted_sub, args):
    lt2 = time.strftime("%Y-%m-%d--%H_%M_%S", time.localtime())

    pkl_file = os.path.join(dump_pickle_dir, f"{args.machine_name}_from_{args.SG_from}_@{lt2}.pkl")
    pickle.dump(
        converted_sub,
        open(pkl_file, "wb"),
    )
    return pkl_file


if __name__ == "__main__":
    main()

