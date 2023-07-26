import random
import string
import subprocess
from typing import Union

from time import time as timer

from dataloaders.loader_utils import timeout, MyTimeoutError

from tokenizer.tokenizerAPI import (
    tokenizerAPI_IN2R,
    tokenizerAPI_IR2N,
    tokenizerAPI_ON2R,
    tokenizerAPI_OR2N,
)



# 🟩 below raw conversion codes copied from apps.py
def convert_raws_to_objs(list_of_raw_str):
    res = []
    # bad_indices = []
    for i, each in enumerate(list_of_raw_str):
        raw_lst = each.strip().split("\n") if isinstance(each, str) else each
        if data_type_is_not_str(raw_lst):
            # bad_indices.append(i)
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



convert_obj_back_to_raw = lambda lst2: '\n'.join([' '.join([str(x) for x in lst1]) for lst1 in lst2])

def convert_io_back_to_raw(io_obj):
    raw_x, raw_y = io_obj
    raw_x = convert_obj_back_to_raw(raw_x)
    raw_y = convert_obj_back_to_raw(raw_y)
    return raw_x, raw_y

def check_io_match_one_sample_int(io_ns, code_ns, sanity_check_timeout):
    io_obj = tokenizerAPI_IN2R(io_ns)
    code = tokenizerAPI_ON2R(code_ns)
    is_match, exec_out, prt_str = check_io_match_one_sample_obj(io_obj, code, sanity_check_timeout)
    return is_match, exec_out, prt_str


def check_io_match_one_sample_obj(io_obj, code, sanity_check_timeout, want_print = False):
    assert type(sanity_check_timeout) is int, 'must provide integer timelimit'

    try:
    # if 1:
        @timeout(sanity_check_timeout)
        def run_():
            raw_x, raw_y = convert_io_back_to_raw(io_obj)
            exec_out, core_exec_time = run_input_print_code(code, raw_x)
            is_match = exec_out==raw_y
            return is_match, exec_out, core_exec_time
        is_match, exec_out, core_exec_time = run_()
    except MyTimeoutError:
    # else:
        is_match, exec_out, core_exec_time = False, RuntimeError(f'Timeout, limit is {sanity_check_timeout}s. NO error though.'), -1
    except:
        is_match, exec_out, core_exec_time = False, RuntimeError(f'Not timeout, other errors.'), -1



    prt_str = f'In check match: \nI/O:\n\t\t{repr(io_obj)}\nExec Result:\n\t\t{repr(exec_out)}\nIs Match:\n\t\t{repr(is_match)} \n core_exec_time:\n\t {core_exec_time}'
    if want_print:
        print(prt_str)
    return is_match, exec_out, prt_str


def check_match_one_instance(instance_io, instance_code, sanity_check_timeout):
    import random
    io_inp = random.choice(instance_io)
    code_inp = random.choice(instance_code)
    is_match, exec_out, io_obj, code = check_io_match_one_sample_int(io_inp, code_inp, sanity_check_timeout)
    return

def forward_run_code(raw_x, code, timelimit):
    try:
        @timeout(timelimit)
        def forward():
            return run_input_print_code(code, raw_x)
        exec_out = forward()
    except:
        exec_out = RuntimeError('Program + idata execution Timeout.')
    return exec_out


def run_input_print_code(code: str, idata: str, debug: bool = False) -> Union[str, int]:
    """
    Runs code and returns output.
    If output is string, code executed properly.
    If output is -1 (int), code failed somewhere.
    :param code: String of code.
    :param idata: String of input to script. Use \n for new line, not list!!!
    :param debug: Bool flag to print error.
    :return: Output of code (or) -1 if failed.
    """
    rnd = "".join(random.choices(string.ascii_letters + string.digits, k=16))
    tmp_file = f"/tmp/input_print_{rnd}.py"
    open(tmp_file, "w").write(code)
    t0 = timer()
    result = subprocess.run(["python", tmp_file], input=idata, capture_output=True, text=True)
    core_exec_time = timer() - t0
    subprocess.run(["rm", tmp_file])

    if len(result.stderr):
        if debug:
            print(result.stderr)
        # return RuntimeError('Program execution yield error; did NOT timeout.'), core_exec_time
        return RuntimeError(f'Program execution procedure error; did NOT timeout. Full error message: \n\n{result.stderr}'), core_exec_time
    else:
        return result.stdout.strip(), core_exec_time

