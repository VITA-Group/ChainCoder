from tqdm import tqdm
import os
import time
import sys
import re

from argparse import ArgumentParser

from dataloaders.apps import get_apps_rawloader
from dataloaders.code_contests import get_contest_rawloader
from dataloaders.check_exec_match import check_io_match_one_sample_obj
from dataloaders.loader_utils import timeout, save_raw
from tokenizer.tokenizerAPI import (
    vocabulary_defs, load_txt,
    tokenizerAPI_OR2T,
    tokenizerAPI_OT2R,
    tokenizerAPI_IT2R,
    tokenizerAPI_IR2T,
)




def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        '--apps_data_root',
        type = str,
        default = '/path/to/apps/APPS',
        help = 'Root of data downloaded from APPS (https://github.com/hendrycks/apps).'
    )
    parser.add_argument(
        '--one_sample_tokenization_timelimit',
        type = int,
        default = 5,
        help = 'Time limit for tokenizing one sample. If exceed, discard this sample. Suggested: 2s is way more than enough, if one sample even exceeds 2s, it may take more than tens of seconds.'
    )
    parser.add_argument(
        '--one_instance_vocab_collection_timelimit',
        type = int,
        default = 5000,
        help = 'If one instance exceed this number and did not finish all samples, it will be discarded and the results not saved.'
    )

    parser.add_argument(
        '--STTD_output_root',
        type = str,
        default = 'some_new_dump_path',
        help = 'This will be the location where all the STTD files are dumped into. Set this to non-existent dir on your machine.'
    )

    args = parser.parse_args()

    return args

args = parse_args()


def add_wrap(code):
    code = code.split('\n')
    code = ['def syntaxformer_added_top_wrap_func():'] + ['    ' + l for l in code] + ['syntaxformer_added_top_wrap_func()\n']
    code = '\n'.join(code)
    return code

def tok_vocab_and_save():
    vocabulary_defs.refuse_unseen_tokens = False

    apps_trainloader = get_apps_rawloader(
        mode="train",
        difficulties=["introductory", "interview", "competition"], 
        apps_data_root=args.apps_data_root
    )
    apps_testloader = get_apps_rawloader(
        mode="test", 
        difficulties=["introductory", "interview", "competition"], 
        apps_data_root=args.apps_data_root
    )
    
    
    contest_train_loader = get_contest_rawloader('train')
    contest_test_loader = get_contest_rawloader('test')


    dataloaders = [
        (apps_testloader, 'apps_test'), 
        (apps_trainloader, 'apps_train'), 
        (contest_train_loader, 'contest_train'),
        (contest_test_loader, 'contest_test'), 
        ]


    instance_id = 0
    (the_wall_dir, vo_dir) = None, None

    for idataloader in range(len(dataloaders)):

        dataloader, which_loader  = dataloaders[idataloader]
        print(f'\n\n\n    Now using dataloader  {which_loader}  \n\n')

        for i, one_instance in enumerate(tqdm(dataloader)):
            if one_instance is None: continue

            try:
                status = ensureTokenization_and_save_raw(which_loader, instance_id, one_instance)
            except:
                print(f'instance error: {which_loader}, loader_output_id = {i}, error is: {sys.exc_info()[:-1]}')
                continue

            if status[0]=='👌':
                instance_id += 1
                (the_wall_dir, vo_dir) = status[1]

            if i%100==10:
                print(f'\n\n 🟨 🟨 checking: \n Now using {which_loader}, idataloader = {idataloader}, this-loader-id = {i} ; Total instances so far =  {instance_id}; vocab dirs = {the_wall_dir, vo_dir}')

        print(f'\n\n 🟨 🟨 🟨 🟨 🟨 🟨 🟨 🟨 \n In step0 finished {which_loader} generated xx instances out of {i} ; Total instances so far =  {instance_id}')
        time.sleep(1)
    return


@timeout(args.one_instance_vocab_collection_timelimit)
def ensureTokenization_and_save_raw(which_loader, instance_id, one_instance):
    global vocabulary_defs
    
    pcodes_raw, pxs_raw, pys_raw, pio_objs, pdescription, pdifficulty = one_instance['codes_raw'], one_instance['xs_raw'], one_instance['ys_raw'], one_instance["io_objs"], one_instance["description"], one_instance["difficulty"]

    vocab_SO_dic = dict(eval(load_txt(vocabulary_defs.VOCAB_SO_FILE)))
    vocab_SI_set = set(eval(load_txt(vocabulary_defs.VOCAB_SI_FILE)))
    vocab_CC_set = set(eval(load_txt(vocabulary_defs.VOCAB_CC_FILE)))

    def update_vocab_code(synSeq_code, contSeq_code):
        global vocabulary_defs
        nonlocal vocab_SO_dic, vocab_SI_set, vocab_CC_set
        for x in synSeq_code:
            if vocabulary_defs.is_unseen_SO(x):
                k = vocabulary_defs.toKey(x, 'O', 'syn', need_update=False)
                vocabulary_defs.update([k], ['O'], ['syn'])
                vocab_SO_dic[x] = 1
            else:
                if x in vocab_SO_dic:
                    vocab_SO_dic[x] += 1
        for x in contSeq_code:
            if vocabulary_defs.is_unseen_CC(x):
                k = vocabulary_defs.toKey(x, 'O', 'cont', need_update=False)
                vocabulary_defs.update([k], ['O'], ['cont'])
                vocab_CC_set.update([x])


    def update_vocab_iodata(synSeq_io, contSeq_io):
        global vocabulary_defs
        nonlocal vocab_SO_dic, vocab_SI_set, vocab_CC_set

        for x in synSeq_io:
            if vocabulary_defs.is_unseen_SI(x):
                vocab_SI_set.update([x])
                k = vocabulary_defs.toKey(x, 'I', 'syn', need_update=False)
                vocabulary_defs.update([k], ['I'], ['syn'])
        for x in contSeq_io:
            if vocabulary_defs.is_unseen_CC(x):
                vocab_CC_set.update([x])
                k = vocabulary_defs.toKey(x, 'I', 'cont', need_update=False)
                vocabulary_defs.update([k], ['I'], ['cont'])


    codes_nameReplaced = []
    codes_raw = []

    # 🟩 ensure tokenization for code
    for code in pcodes_raw:



        try:
            @timeout(args.one_sample_tokenization_timelimit)
            def run_code_tokenization(code):
                is_match, exec_out, prt_str = check_io_match_one_sample_obj(pio_objs[0], code, sanity_check_timeout=1) # only check if there's return issue
                

                errmsg = str(exec_out)
                return_err_locs = re.findall(r'SyntaxError(.*)return(.*)outside function', errmsg)
                if len(return_err_locs)!=0:
                    code = add_wrap(code)
                    

                synSeq_code, contSeq_code = tokenizerAPI_OR2T(code)
                return synSeq_code, contSeq_code
            synSeq_code, contSeq_code = run_code_tokenization(code)
            update_vocab_code(synSeq_code, contSeq_code)
            name_replaced_recov_code_str = tokenizerAPI_OT2R(synSeq_code, contSeq_code)
        except:
            print('CODE tokenization error, Discarded:\n', sys.exc_info()[:-1])
            continue

        if synSeq_code==[]:
            print('in step0, tokenization fail (too many diy names), discarded this CODE, which is:')
            print(code)
            continue

        codes_nameReplaced.append(name_replaced_recov_code_str)
        codes_raw.append(code)

    xs_raw = []
    ys_raw = []
    io_objs = []  # shape: [sample, (input, output) tuple, token dim]
    for x_raw, y_raw, io_2t in zip(pxs_raw, pys_raw, pio_objs):


        try:
            @timeout(args.one_sample_tokenization_timelimit)
            def run_possibly_super_long_tokenization(io_2t):
                synSeq_io, contSeq_io = tokenizerAPI_IR2T(io_2t)
                update_vocab_iodata(synSeq_io, contSeq_io)
                rev = tokenizerAPI_IT2R(synSeq_io, contSeq_io)
                return rev, synSeq_io, contSeq_io
            rev, synSeq_io, contSeq_io = run_possibly_super_long_tokenization(io_2t)

        except:
            print('SAMPLE tokenization error, Discarded.')
            continue

        if rev!=io_2t:
            print('in step0, io tokenization fail, discarded this SAMPLE, which is:')
            print(io_2t)
            continue

        else:

            io_objs.append(io_2t)
            xs_raw.append(x_raw)
            ys_raw.append(y_raw)


    if len(io_objs)<=2 or len(codes_nameReplaced)==0:
        print('in step0, valid sample num too small, discarded this INSTANCE.')
        return '😭', (None, None)


    # 🟩 save raw files

    raw_dir = os.path.join(args.STTD_output_root, f'difficulty_{pdifficulty}')
    os.makedirs(raw_dir, exist_ok=True)

    cross_samp_join = '\n\n# 🟨 🟨 🟨 🟨 \n\n'
    codes_readable_raw = cross_samp_join.join(codes_raw)
    codes_readable_nameReplaced = cross_samp_join.join(codes_nameReplaced)
    iodatas_readable = cross_samp_join.join([repr(tuple(x)) for x in io_objs])
    save_raw(raw_dir, which_loader, instance_id, 
        codes_raw, codes_nameReplaced, codes_readable_raw, codes_readable_nameReplaced,
        xs_raw, ys_raw, io_objs, iodatas_readable, 
        pdescription)

    # 🟩 save vocabs
    print(vocab_SO_dic, file=open(vocabulary_defs.VOCAB_SO_FILE, 'w'))
    print(vocab_SI_set, file=open(vocabulary_defs.VOCAB_SI_FILE, 'w'))
    print(vocab_CC_set, file=open(vocabulary_defs.VOCAB_CC_FILE, 'w'))
    the_wall_dir = vocabulary_defs.save_the_great_wall()
    vo_dir = vocabulary_defs.VOCAB_SO_FILE

    return '👌', (the_wall_dir, vo_dir)


if __name__ == '__main__':

    tok_vocab_and_save()


