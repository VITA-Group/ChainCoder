import json
import os
from argparse import ArgumentParser
from glob import glob
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np

VERBOSE = 1

from dataloaders.check_exec_match import run_input_print_code




def convert_raws_to_objs(list_of_raw_str):
    res = []
    for i, each in enumerate(list_of_raw_str):
        raw_lst = each.strip().split("\n") if isinstance(each, str) else each
        if data_type_is_not_str(raw_lst):
            obj = None
        else:
            obj = convert_lines_to_obj(raw_lst)
        res.append(obj)
    return res

def try_convert_number(n):
    # input is a string.
    def number_is_int(n):
        if n[0] in ['+', '-']:
            return n[1:].isdigit()
        else:
            return n.isdigit()

    is_number = True
    try:
        num = float(n)
        # check "nan" 
        is_number = (num == num)   # nan should return False
    except ValueError:
        is_number = False
    

    if is_number:
        if number_is_int(n):
            obj = int(n)
        else:
            obj = float(n)
    else:
        obj = n
    return obj

def data_type_is_not_str(lst_1D):
    if type(lst_1D) is not list:
        return True
    for s_space in lst_1D:
        if type(s_space) is not str:
            return True
    return False


def convert_lines_to_obj(lst_1D):
    lst_obj = []
    if type(lst_1D) is not list:
        raise ValueError
    for s_space in lst_1D:
        if type(s_space) is str:
            s_lst = s_space.split()
            for i, s in enumerate(s_lst):

                sobj = try_convert_number(s)

                s_lst[i] = sobj
            lst_obj.append(s_lst)
        else:
            raise ValueError
            lst_obj.append(s_space)
    return lst_obj



class APPS(Dataset):
    def __init__(self, mode, difficulties, apps_data_root):
        """
        Args:
            modes: train, test
            difficulty: introductory interview competition
        """
        all_instances = glob(os.path.join(apps_data_root, mode, '**'))
        self.instances = list(
            filter(lambda i: json.load(open(os.path.join(i, 'metadata.json')))["difficulty"] in difficulties, all_instances)
        )


    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        try:
            codes = json.load(open(os.path.join(self.instances[idx], 'solutions.json')))
            iodata = json.load(open(os.path.join(self.instances[idx], 'input_output.json')))
            description = open(os.path.join(self.instances[idx], 'question.txt')).read().split("-----Input-----")[0].strip()
            meta = json.load(open(os.path.join(self.instances[idx], "metadata.json")))
        except FileNotFoundError as e:
            print(f'APPS file not found: {str(e)}.')
            return None

        check_io_match_now = False
        if check_io_match_now:
            for each_code in codes:
                try:
                    list_str_out = run_input_print_code(each_code, x_raw[0])
                except:
                    print('in apps dataloader, exec bug, discarded this SAMPLE.')
                    continue

                if list_str_out!=y_raw:
                    print('in apps dataloader, encountered wrong dataset label, discarded this SAMPLE.')
                    continue

        # sometimes inputs and outputs are having an extra first dimension (equivalent of unsqueeze(0))
        # below is basically a squeeze(0) operation
        if len(iodata["inputs"]) == 1 and isinstance(iodata["inputs"][0], list) and len(iodata["inputs"][0]) > 0:
            iodata["inputs"] = iodata["inputs"][0]
        if len(iodata["outputs"]) == 1 and isinstance(iodata["outputs"][0], list) and len(iodata["outputs"][0]) > 0:
            iodata["outputs"] = iodata["outputs"][0]

        # sometimes xs_raw and ys_raw have string, and sometimes directly a list
        # so we have an if else in those lines below
        xs_raw = [[each.strip().split("\n") if isinstance(each, str) else each] for each in iodata["inputs"]]
        ys_raw = [[each.strip().split("\n") if isinstance(each, str) else each] for each in iodata["outputs"]]
        # if there's no group (x,y), below codes are equal to: x_objs = convert_raws_to_objs(iodata["outputs"]) then remove None jointly



        io_objs = []
        for x_raw, y_raw in zip(xs_raw, ys_raw):
            assert len(x_raw)==1
            assert len(y_raw)==1

            x = x_raw[0]    # 1-D list, elem is string, but contain space; space should be further splited. e.g.: ['2 3', 'abc 4', 'd']
            y = y_raw[0]

            if data_type_is_not_str(x) or data_type_is_not_str(y):
                if VERBOSE:
                    print(f'in APPS, non-standard iodata, dropped x, x, which is:\n{repr(x)}\n{repr(y)}')
                continue

            x_obj = convert_lines_to_obj(x)   # supposed to be 2-D list, final shape: [each line, obj in line after split and eval]  e.g.: [[2, 3], ['abc', 4], ['d']]
            y_obj = convert_lines_to_obj(y)

            io_objs.append([x_obj, y_obj])

        pcodes = codes
        codes = []
        for code in pcodes:
            if 'input(' in code:
                codes.append(code)
        

        if len(codes)==0 or len(io_objs)<=1:
            print('In APPS, valid regular data number too small, discard this problem.')
            return None


        samples = {
            "codes_raw": codes,
            "xs_raw": xs_raw,
            "ys_raw": ys_raw,
            "io_objs": io_objs,

            "description": description,
            "difficulty": meta['difficulty'],
            "info": {},
        }

        try:
            assert len(samples['xs_raw'])==len(samples['ys_raw'])==len(samples["io_objs"]), (len(samples['xs_raw']), len(samples['ys_raw']), len(samples["io_objs"]))
            return samples
        except:
            print(f"in apps data, data len not equal: {len(samples['xs_raw']), len(samples['ys_raw']), len(samples['io_objs'])}")
            return None



def get_apps_rawloader(mode, difficulties, apps_data_root):
    def process(batch):
        assert len(batch)==1
        return batch[0]

    dataset = APPS(mode, difficulties, apps_data_root)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=process)

    return dataloader




def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--mode", default="test", choices=["train", "test"])
    parser.add_argument("--difficulty", default="introductory interview competition average", nargs="+")
    args = parser.parse_args()

    return args


def test_APPS_rawloader():
    args = parse_args()

    args.difficulty = args.difficulty.split(" ")

    dataloader = get_apps_rawloader(args.mode, args.difficulty)
    print(len(dataloader))
    for i in dataloader:
        print(i)

    total = 0
    for i in tqdm(dataloader):
        assert len(i["code_strings"]) == len(i["xs_raw"]) == len(i["ys_raw"])
        total += len(i["code_strings"])
    print(total)


if __name__ == "__main__":
    test_APPS_rawloader()
