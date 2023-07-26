from glob import glob
import numpy as np
import os

import errno
import signal
from functools import wraps, partial
import math
import time


def load_txt(fname, as_str=True):
    x = []
    with open(fname, 'r') as f:
        for line in f.readlines():
            if line[-1]=='\n':
                x.append(line[:-1])
            else:
                x.append(line)
    if as_str:
        x = '\n'.join(x)
    return x


def load_all_instances(subfolder, file_id_re='*', shuffle=True):
    name_to_int = lambda i: int(i.split("_")[-1].split('.')[0])

    if len(glob(os.path.join(subfolder, f'*codes_nameReplaced*{file_id_re}.py')))==0:
        return [[] for _ in range(7)]

    def glob_sorted(restr):
        restr = os.path.join(subfolder, restr)
        files = glob(restr)
        indices = list(map(name_to_int, files))

        files = sorted(zip(files, indices), key=lambda x:x[1])
        return list(zip(*files))
    codes_nameReplaced, ind_cre = glob_sorted(f'*codes_nameReplaced*{file_id_re}.py')
    codes_raw, ind_crw = glob_sorted(f'*codes_raw*{file_id_re}.py')
    xs_raw, ind_xr = glob_sorted(f'*xs_raw*{file_id_re}.py')
    ys_raw, ind_yr = glob_sorted(f'*ys_raw*{file_id_re}.py')
    iodatas_obj, ind_o = glob_sorted(f'*io_objs*{file_id_re}.py')
    descriptions, ind_d = glob_sorted(f'*description*{file_id_re}.txt')
    file_names, ind_fn = glob_sorted(f"*codes_raw*{file_id_re}.py")

    try:
        assert len(codes_raw)==len(codes_nameReplaced)==len(xs_raw)==len(ys_raw)==len(iodatas_obj)==len(descriptions)==len(file_names)
    except:
        print('file missing due to scp or generation! num of instances of each file types = ', (len(codes_raw),len(codes_nameReplaced),len(xs_raw),len(ys_raw),len(iodatas_obj),len(descriptions), len(file_names)), 'Now dropped the missing instance.')
        def drop_missing():
            
            _codes_raw, _codes_nameReplaced, _xs_raw, _ys_raw, _iodatas_obj, _descriptions, _fnames = [[] for _ in range(7)]
            x2i_cre = {x:i for i, x in enumerate(ind_cre)}
            x2i_crw = {x:i for i, x in enumerate(ind_crw)}
            x2i_xr = {x:i for i, x in enumerate(ind_xr)}
            x2i_yr = {x:i for i, x in enumerate(ind_yr)}
            x2i_o = {x:i for i, x in enumerate(ind_o)}
            x2i_d = {x:i for i, x in enumerate(ind_d)}
            x2i_fn = {x:i for i, x in enumerate(ind_fn)}
            set_ind_cre, set_ind_crw, set_ind_xr, set_ind_yr, set_ind_o, set_ind_d, set_int_fn = set(ind_cre), set(ind_crw), set(ind_xr), set(ind_yr), set(ind_o), set(ind_d), set(ind_fn)
            inters = set_ind_cre.intersection(set_ind_crw).intersection(set_ind_xr).intersection(set_ind_yr).intersection(set_ind_o).intersection(set_ind_d).intersection(set_int_fn)

            for x in inters:
                _codes_raw.append(codes_raw[x2i_cre[x]])
                _codes_nameReplaced.append(codes_nameReplaced[x2i_crw[x]])
                _xs_raw.append(xs_raw[x2i_xr[x]])
                _ys_raw.append(ys_raw[x2i_yr[x]])
                _iodatas_obj.append(iodatas_obj[x2i_o[x]])
                _descriptions.append(descriptions[x2i_d[x]])
                _fnames.append(file_names[x2i_fn[x]])

            return _codes_raw, _codes_nameReplaced, _xs_raw, _ys_raw, _iodatas_obj, _descriptions, _fnames
        codes_raw, codes_nameReplaced, xs_raw, ys_raw, iodatas_obj, descriptions, file_names = drop_missing()
        name_to_int_ls = lambda ls: [name_to_int(x) for x in ls]
        assert set(name_to_int_ls(codes_raw)) == set(name_to_int_ls(codes_nameReplaced)) == set(name_to_int_ls(xs_raw)) == set(name_to_int_ls(ys_raw)) == set(name_to_int_ls(iodatas_obj)) == set(name_to_int_ls(descriptions)) == set(name_to_int_ls(file_names))



    if len(codes_raw)==0:
        print("empty samples")
        return [[] for _ in range(6)]


    # 🟩 load files
    batch_load = lambda flst: [load_txt(f) for f in flst]
    codes_raw, codes_nameReplaced, xs_raw, ys_raw, iodatas_obj, descriptions = batch_load(codes_raw), batch_load(codes_nameReplaced), batch_load(xs_raw), batch_load(ys_raw), batch_load(iodatas_obj), batch_load(descriptions)

    # 🟩 batch eval
    def replace_too_large_int_with_inf(io_strings):
        new_strings = []
        max_digits = 4300
        for string in io_strings:
            res = ''
            prev_group = ''
            for s in string:
                if not s.isdigit():
                    if len(prev_group)>=max_digits:
                        prev_group = 'inf'
                    res += prev_group
                    res += s
                    prev_group = ''
                else:
                    prev_group += s
            if len(prev_group)>=max_digits:
                res += 'inf'
            else:
                res += prev_group
            new_strings.append(res)
        return new_strings
    iodatas_obj = replace_too_large_int_with_inf(iodatas_obj)



    batch_eval = lambda objlst: [evalio(x) for x in objlst]
    codes_raw, codes_nameReplaced, xs_raw, ys_raw, iodatas_obj = batch_eval(codes_raw), batch_eval(codes_nameReplaced), batch_eval(xs_raw), batch_eval(ys_raw), batch_eval(iodatas_obj)



    
    # 🟩 shuffle jointly
    if shuffle:
        perm = np.random.permutation(len(codes_raw))
        shuffle = lambda lst: [lst[i] for i in perm]
        codes_raw, codes_nameReplaced, xs_raw, ys_raw, iodatas_obj, descriptions, file_names = shuffle(codes_raw), shuffle(codes_nameReplaced), shuffle(xs_raw), shuffle(ys_raw), shuffle(iodatas_obj), shuffle(descriptions), shuffle(file_names)


    return codes_raw, codes_nameReplaced, xs_raw, ys_raw, iodatas_obj, descriptions, file_names

def evalio(x):
    return eval(x, {'inf': float('inf'), 'nan': float('nan')})


def parse_loadername_from_filename(filename):
    basename = os.path.basename(filename)
    loadername = '_'.join(basename.split('_')[:2])
    inst_id_orig = basename.split('_')[-1][:-3]
    return loadername, inst_id_orig

def save_raw(_dir, which_loader, instance_id, 
        codes_raw, codes_nameReplaced, codes_readable_raw, codes_readable_nameReplaced,
        xs_raw, ys_raw, io_objs, iodatas_readable,
        description):

    os.makedirs(_dir, exist_ok=True)
    if type(instance_id) is int:
        instance_id = f'{instance_id:06d}'
    print(codes_raw, file=open(os.path.join(_dir, f'{which_loader}_codes_raw_{instance_id}.py'), 'w'))
    print(codes_nameReplaced, file=open(os.path.join(_dir, f'{which_loader}_codes_nameReplaced_{instance_id}.py'), 'w'))
    print(xs_raw, file=open(os.path.join(_dir, f'{which_loader}_xs_raw_{instance_id}.py'), 'w'))
    print(ys_raw, file=open(os.path.join(_dir, f'{which_loader}_ys_raw_{instance_id}.py'), 'w'))
    print(io_objs, file=open(os.path.join(_dir, f'{which_loader}_io_objs_{instance_id}.py'), 'w'))
    print(description, file=open(os.path.join(_dir, f'{which_loader}_description_{instance_id}.txt'), 'w'))

    os.makedirs(os.path.join(_dir, 'readable'), exist_ok=True)
    print(codes_readable_raw, file=open(os.path.join(_dir, 'readable', f'{which_loader}_codes_readable_raw_{instance_id}.py'), 'w'))
    print(codes_readable_nameReplaced, file=open(os.path.join(_dir, 'readable', f'{which_loader}_codes_readable_nameReplaced_{instance_id}.py'), 'w'))
    print(iodatas_readable, file=open(os.path.join(_dir, 'readable', f'{which_loader}_iodatas_readable_{instance_id}.py'), 'w'))
    print(description, file=open(os.path.join(_dir, 'readable', f'{which_loader}_description_{instance_id}.txt'), 'w'))





def shuffled(iterable):
    lst = list(iterable)
    np.random.shuffle(lst)
    return lst

class MyTimeoutError(BaseException):
    pass
def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(repeat_id, signum, frame):
            signal.signal(signal.SIGALRM, partial(_handle_timeout, repeat_id + 1))
            signal.alarm(seconds)
            raise MyTimeoutError(error_message)

        def wrapper(*args, **kwargs):
            old_signal = signal.signal(signal.SIGALRM, partial(_handle_timeout, 0))
            old_time_left = signal.alarm(seconds)
            assert type(old_time_left) is int and old_time_left >= 0
            if 0 < old_time_left < seconds:  # do not exceed previous timer
                signal.alarm(old_time_left)
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
            finally:
                if old_time_left == 0:
                    signal.alarm(0)
                else:
                    sub = time.time() - start_time
                    signal.signal(signal.SIGALRM, old_signal)
                    signal.alarm(max(0, math.ceil(old_time_left - sub)))
            return result

        return wraps(func)(wrapper)

    return decorator
