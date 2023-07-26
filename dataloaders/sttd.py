import itertools
import os
import pickle
from glob import glob
import numpy as np
from torch.utils.data import DataLoader, Dataset

VERBOSE = 0

class STTTDDataset(Dataset):
    def __init__(self, params, pickle_root):
        if params.only_do_subfolder != '':
            subfolders = [params.only_do_subfolder]
        else:
            subfolders = ['difficulty_introductory', 'difficulty_interview', 'difficulty_dm_code_contest']
        load_pkl_from = [os.path.join(pickle_root, x, '*.pkl') for x in subfolders]

        print(f'\n\nIn STTD dataloader, loading from: {load_pkl_from}')

        all_existing_files = list(itertools.chain.from_iterable([glob(x) for x in load_pkl_from]))
        if params.most_recent_pickle_num == 'all':
            self.files = all_existing_files
            print(f'When init dataloader, grabbing all {len(all_existing_files)} existing pickles.')
        else:
            most_recent_pickle_num = int(params.most_recent_pickle_num)
            self.files = sorted(all_existing_files, key=os.path.getmtime)[-most_recent_pickle_num:]
            print(f'When init dataloader, grabbing {most_recent_pickle_num} most recent pickles out of {len(all_existing_files)}.')
        assert len(self.files)>0, f'dataset empty: {load_pkl_from}'
        self.max_nlp_seq_len = 512

        self.max_io_seq_len, self.max_code_seq_len = params.max_io_seq_len, params.max_code_seq_len
        self.batch_size = params.batch_size
        self.samples_per_instance_io = params.samples_per_instance_io
        self.samples_per_instance_code = params.samples_per_instance_code
        self.samples_per_instance_io_hold = params.samples_per_instance_io_hold

        demo = pickle.load(open(self.files[0], "rb"))
        [valid_iodatas_int_is, valid_codes_int_is, valid_desc_distilbert, valid_desc_bert] = demo
        self.instances_per_file = len(valid_iodatas_int_is)

        self.inter_file_order = None
        self.intra_file_order = None
        self.inter_index = 0
        self.intra_index = 0
        self.inter_flag = 0
        self.inter_flag = 0
        self.file_in_memory = False
        self.batch_sampler()

    def __len__(self):
        return len(self.files) * (self.instances_per_file // self.batch_size)

    def batch_sampler(self):
        self.inter_file_order = np.random.permutation(np.arange(len(self.files)))
        assert self.instances_per_file / self.batch_size>0, (self.instances_per_file , self.batch_size)
        self.intra_file_order = [
            np.random.permutation(np.arange(int(self.instances_per_file / self.batch_size)))
            for _ in range(len(self.files))
        ]
        self.inter_index = 0
        self.intra_index = 0
        self.file_in_memory = False


    def __getitem__(self, item):
        if not self.file_in_memory:
            self.file_in_memory = pickle.load(open(self.files[self.inter_file_order[self.inter_index]], "rb"))
            self.file_in_memory[0], bad_insts1 = drop_token_len_exceeds(self.file_in_memory[0], self.max_io_seq_len)
            self.file_in_memory[1], bad_insts2 = drop_token_len_exceeds(self.file_in_memory[1], self.max_code_seq_len)
            bad_insts = set(bad_insts1+bad_insts2)
            if len(bad_insts)>0:
                for itp in range(4):  # if bad, pad ...
                    def drop_certain_indices(lst, indices):
                        new = []
                        for i, x in enumerate(lst):
                            if i not in indices:
                                new.append(x)
                        return new
                    self.file_in_memory[itp] = drop_certain_indices(self.file_in_memory[itp], bad_insts)
                if VERBOSE:
                    print(f'Dataloader Find {len(bad_insts)} bad_insts (tok len too long) out of {self.instances_per_file}... padded with duplication.')
                good_ists = np.random.choice(len(self.file_in_memory[0]), len(bad_insts))
                for itp in range(4):  # if bad, pad ...
                    self.file_in_memory[itp].extend([self.file_in_memory[itp][ig] for ig in good_ists])

        
        idx = self.intra_file_order[self.inter_file_order[self.inter_index]][self.intra_index]
        batch_content = [self.file_in_memory[type_dir][idx : idx + self.batch_size] for type_dir in range(4)]
        batch_content = list(zip(*batch_content))

        data = dict()

        ioData_ist2_all = list(map(lambda i: list(map(lambda ii: ii, i[0])), batch_content))
        codes_ist2_in_file = list(map(lambda i: i[1], batch_content))

        def select_k_samples(ist2, release_hold):
            ist2_rel, ist2_hold = [], []

            for i, st2 in enumerate(ist2):
                num_samp = len(st2)
                if num_samp >= sum(release_hold):
                    replace = False
                else:
                    if VERBOSE:
                        print(f'In dataloader, requested too many samples for I/O: samples_per_instance_io + samples_per_instance_io_hold > total samples: sum({release_hold}) > {num_samp}. Returned samples now have duplications!')
                    replace = True
                idsall = np.random.choice(num_samp, sum(release_hold), replace=replace)
                ids_rel, ids_hold = idsall[:release_hold[0]], idsall[release_hold[0]:]#idsall[-release_hold[1]:]
                ist2_rel.append([st2[j] for j in ids_rel])
                ist2_hold.append([st2[j] for j in ids_hold])
            return ist2_rel, ist2_hold

        data["ioData_ist2"], data["ioData_ist2_holdout"] = select_k_samples(ioData_ist2_all, [self.samples_per_instance_io, self.samples_per_instance_io_hold])

        data["desc_distilbert_it"] = list(map(lambda i: i[2]['input_ids'], batch_content))
        data["desc_bert_it"] = list(map(lambda i: i[3]['input_ids'], batch_content))
        data["desc_distilbert_it"] = [x[:self.max_nlp_seq_len] for x in data["desc_distilbert_it"]]
        data["desc_bert_it"] = [x[:self.max_nlp_seq_len] for x in data["desc_bert_it"]]


        codes_ist2 = []
        for inst_id in range(self.batch_size):
            codes_ist2.append(select_n_from_group(codes_ist2_in_file[inst_id], self.samples_per_instance_code, 'code-sample'))

        data["program_sit2"] = list(zip(*codes_ist2))


        self.intra_index += 1
        if self.intra_index == int(self.instances_per_file / self.batch_size):
            self.file_in_memory = None
            self.intra_index = 0
            self.inter_index += 1
        if self.inter_index == len(self.files):
            self.batch_sampler()

        return data

def drop_token_len_exceeds(ist2, max_len):
    new = []
    for st2 in ist2:
        new.append(list(filter(lambda t2: len(t2)<=max_len, st2)))
    bad_insts = np.where([len(x)==0 for x in new])[0].tolist()
    return new, bad_insts


def get_ChainCoder_dataloader(params, pickle_root, return_dataset=False):

    def collate_fn(batch):
        assert len(batch)==1
        newbatch = batch[0]
        if newbatch==None:
            return None
        coarse = lambda lst: [(lst[:1] if len(lst) >= 1 else lst) + [lst[i] for i in range(len(lst)) if (i+1) % 500 == 0]] + lst # coarse subsequence inserted to index-0 of original [[S3, S4], ...] list. 
        newbatch['program_sit2'] = [list(map(coarse, inner_lst)) for inner_lst in newbatch['program_sit2']]
        return newbatch

    dataset = STTTDDataset(params, pickle_root)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    if return_dataset:
        return dataloader, dataset
    else:
        return dataloader

def pretrain_loader(params):
    from datasets import load_dataset
    dataset = load_dataset("codeparrot/codeparrot-clean", split='train')
    from tokenizer.tokenizerAPI import tokenizerAPI_OR2N

    def collate_fn(batch):
        
        ioData_ist2 = []
        desc_it = []


        program_it2 = []
        for data in batch:
            code_str = data['content']
            print(data['path'])
            desc_it.append('')

            code_int_t2 = tokenizerAPI_OR2N(code_str)
            if code_int_t2==[]:
                return None
            program_it2.append(code_int_t2)

        program_sit2 = [program_it2]

        return {
            'ioData_ist2': ioData_ist2,
            'program_sit2': program_sit2,
            'desc_bert_it': desc_it,
            'desc_distilbert_it': desc_it,
        }


    dataloader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader



def select_n_from_group(group, n, msg=''):
    if len(group) >= n:
        replace = False
    else:
        if VERBOSE:
            print(f'In ASTer dataloader - {msg}, requested > existing:  {n} > {len(group)}. Returned samples now have duplications!')
        replace = True
    indices = np.random.choice(len(group), n, replace=replace)
    selected = [group[i] for i in indices]
    return selected

