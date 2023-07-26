import os
from joblib import Parallel, delayed

from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm






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
        # return [lst_1D]
    for s_space in lst_1D:
        if type(s_space) is not str:
            return True
    return False


def convert_lines_to_obj(lst_1D):
    lst_obj = []
    if type(lst_1D) is not list:
        raise ValueError
    #     # return [lst_1D]
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


def get_contest_rawloader(split="train", datapath="~/.cache/huggingface/datasets"):
    dataset = load_dataset("deepmind/code_contests", split=split, cache_dir=datapath)

    def preprocess(batch):
        assert len(batch)==1
        each = batch[0]

        samples = dict()
        codes = [soln for lang, soln in zip(each["solutions"]["language"], each["solutions"]["solution"]) if lang == 3]

        xs_raw = [eachh.strip().split("\n") for eachh in each["public_tests"]["input"] + each["generated_tests"]["input"]]
        ys_raw = [eachh.strip().split("\n") for eachh in each["public_tests"]["output"] + each["generated_tests"]["output"]]
        description = each["description"]


        io_objs = []
        for x_raw, y_raw in zip(xs_raw, ys_raw):
            x = x_raw    # 1-D list, elem is string, but contain space; space should be further splited. e.g.: ['2 3', 'abc 4', 'd']
            y = y_raw

            if data_type_is_not_str(x) or data_type_is_not_str(y):
                # print('in dm contest, non-standard iodata, dropped.')
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
            print('In dm contest, valid regular data number too small, discard this problem.')
            return None


        samples = {
            "codes_raw": codes,
            "xs_raw": xs_raw,
            "ys_raw": ys_raw,
            "io_objs": io_objs,

            "description": description,
            "difficulty": 'dm_code_contest',
            "info": {},
        }

        try:
            assert len(samples['xs_raw'])==len(samples['ys_raw'])==len(samples["io_objs"]), (len(samples['xs_raw']), len(samples['ys_raw']), len(samples["io_objs"]))
            return samples
        except:
            print(f"in code contest loader, data len not equal: {len(samples['xs_raw']), len(samples['ys_raw']), len(samples['io_objs'])}")
            return None





    dataloader = DataLoader(dataset, batch_size=1, collate_fn=preprocess)
    return dataloader


def save_codes_dump():
    dataset = load_dataset("deepmind/code_contests", split="train")
    os.makedirs("codes_dump", exist_ok=True)
    for each_datapoint in dataset:
        python_codes = [soln for lang, soln in zip(each_datapoint["solutions"]["language"], each_datapoint["solutions"]["solution"]) if lang == 3]
        for each_code in python_codes:
            i = 0
            while os.path.exists(f"codes_dump/sample-{i:04d}.py"):
                i += 1
            if i > 9999:
                return
            open(f"codes_dump/sample-{i:04d}.py", "w").write(each_code)

