# ---- vocabulary definition folder: make sure you can have `os.path.exists(VOCAB_SO_FILE)==True` here. If not, try to use an absolute path for `VOCAB_ROOT`. This is crutial for our code to run. Only when you are generating a new vocabulary set can you have `os.path.exists(VOCAB_SO_FILE)==False`.

VOCAB_ROOT = 'VOCAB_22E4'
VOCAB_SO_FILE = f'{VOCAB_ROOT}/code_syntax_vocabulary.json'
VOCAB_SI_FILE = f'{VOCAB_ROOT}/iodata_syntax_vocabulary.txt'
VOCAB_CC_FILE = f'{VOCAB_ROOT}/wild_content_vocabulary.txt'
VOCAB_GREAT_WALL_FILE = f'{VOCAB_ROOT}/the_great_wall.txt'





# REFUSE_UNSEEN_TOKENS = False
REFUSE_UNSEEN_TOKENS = True
robust_tokenization_after_fix_vocab = True




import copy
import ast
from six.moves import cStringIO
import json
import numpy as np
from collections import defaultdict
import itertools
import os

from tokenizer import astunparse, python_syntax
from tokenizer.python_syntax import *






os.makedirs(VOCAB_ROOT, exist_ok=True)
if REFUSE_UNSEEN_TOKENS:
    warning_msg = f'The VOCAB_ROOT in tokenization_algorithm.py seems not properly set, as no file is found in: {VOCAB_SO_FILE}. \nIn aster/src/tokenizer/tokenization_algorithm.py:\n\t🟨 if "REFUSE_UNSEEN_TOKENS" is set to True, the "VOCAB_ROOT" should be manually configured, and should contain the four pre-generated vocabulary files; \n\t🟨 if "REFUSE_UNSEEN_TOKENS" is set to False, the "VOCAB_ROOT" shall better be an non-existent dir, and this case shall only happen during the vocabulary collection (pre-generation) phase.'



    assert os.path.exists(VOCAB_SO_FILE), warning_msg
    assert len(os.listdir(VOCAB_ROOT))==4 and 'the_great_wall.txt' in os.listdir(VOCAB_ROOT), warning_msg

else:
    if not os.path.exists(VOCAB_SO_FILE):
        print(f'Previous vocab files does not exist in {VOCAB_SO_FILE}, establishing new ones: {VOCAB_SO_FILE}  \n ❗️ ❗️ ❗️ \n  This msg only appear when you are sweeping the dataset to generate the vocabulary, and should ONLY appear ONCE!\n\n\t Seems you are trying to generate new vocabulary into: {VOCAB_ROOT}. \n\n')
        print(dict(), file=open(VOCAB_SO_FILE, 'w'))
        print(set(), file=open(VOCAB_SI_FILE, 'w'))
        print(set(), file=open(VOCAB_CC_FILE, 'w'))

# VERBOSE = False
VERBOSE = True
MAX_INT_TOKEN = 200
MAX_DIY_NAME_TOKEN = 50







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
def get_all_definitions(module):
    all_defined = []
    for x in dir(module):
        if not x.startswith('__'):
            all_defined.append(x)
    name2def = {x: eval(x) for x in all_defined}
    return name2def






def get_vocabulary_defs():
    # 🟩 There are 3 types of vocabs:
        # SI: iodata, syntax
        # SO: program data, syntax
        # CC: content (both io and program)

    def assert_two_set_equal(s1,s2):
        assert len(s1)==len(s2)
        l1 = sorted(s1, key=lambda x: repr(x))
        l2 = sorted(s2, key=lambda x: repr(x))

        while l1:
            p1,p2 = l1.pop(), l2.pop()
            assert p1==p2
        return
    def convert_lists_to_those_dicts(li,lo):
        def list_to_dics(ls):
            t2i = {t:i for i,t in enumerate(ls)}
            i2t = {i:t for i,t in enumerate(ls)}
            return t2i, i2t
        t2i_i,  i2t_i  = list_to_dics(li)
        t2i_o,  i2t_o  = list_to_dics(lo)
        return (
            t2i_i,  i2t_i,
            t2i_o,  i2t_o
        )
    def obj_is_syntax_class(obj):
        return "python_syntax." in str(type(obj))
    def cls_is_syntax_class(cls):
        return "python_syntax." in str(cls)
    def empty_unexpected():
        return {
            'I': 
                {'syn': [], 'cont': []},
            'O':
                {'syn': [], 'cont': []}
            }
    def get_vocab_syntax_code(saved_so):
        syn_token_min_repeat = -1  # set to -1 means inf: keep all syntax tokens
        synVocab2cnt_all = eval(load_txt(saved_so))
        cnt2vocab = defaultdict(list)
        for voc,num in synVocab2cnt_all.items():
            cnt2vocab[num].append(voc)

        plot_dist = 0
        if plot_dist:
            vocabs_cnts = list(synVocab2cnt_all.values())
            print(vocabs_cnts, file=open('_tmp_vocab_cnt.py', 'w'))
        else:
            _voca_num = len(list(synVocab2cnt_all.values()))
            assert _voca_num==212892, f'The vocabulary size for code syntax that you used is {_voca_num} instead of 212892. Replace that number in the assertion here if you believe it is correct - it is sanity check.'

        if syn_token_min_repeat > 0:
            main_cnt2vocab = {k: v for k, v in cnt2vocab.items() if k>syn_token_min_repeat}
            main_vocab_list = list(itertools.chain.from_iterable(main_cnt2vocab.values()))
        else:
            main_cnt2vocab = {k: v for k, v in cnt2vocab.items()}
            main_vocab_list = list(itertools.chain.from_iterable(main_cnt2vocab.values()))
        return main_vocab_list, synVocab2cnt_all
    def grab_content_wild(saved_cc):
        wild_cont_if_any = load_txt(saved_cc).strip()
        if not wild_cont_if_any:
            return []
        return list(eval(wild_cont_if_any))
    def grab_unexpected(saved_unexp):
        prev_unexpected = json.load(open(saved_unexp, 'r'))
        si = prev_unexpected['I']['syn']
        so = prev_unexpected['O']['syn']
        cc = prev_unexpected['I']['cont'] + prev_unexpected['O']['cont']
        return si, so, cc




    # 🟩 Now begin loading syntax vocabulary
    if os.path.exists(VOCAB_SI_FILE):
        vo = list(eval(load_txt(VOCAB_SI_FILE)))
        vocab_SI_saved = list(eval(load_txt(VOCAB_SI_FILE)))
        vocab_SO_saved, synVocab2cnt_all = get_vocab_syntax_code(VOCAB_SO_FILE)
        vocab_CC_wild = grab_content_wild(VOCAB_CC_FILE)
    else:
        print(f'❗️ ❗️ in parse_algorithm.py, when loading vocab, {VOCAB_SI_FILE} does not exist! Now initializing all 4 vocab files. Should ONLY see this when ru dataloader step0.')
        vocab_SI_saved = []
        vocab_SO_saved = []
        vocab_CC_wild = []




    vocab_SO_NLP_syntax_markings = ['<|NLP_EMBEDDER_SYNTAX_MARKING #0|>', '<|NLP_EMBEDDER_SYNTAX_MARKING #1|>', '<|NLP_EMBEDDER_SYNTAX_MARKING #2|>', '<|NLP_EMBEDDER_SYNTAX_MARKING #3|>']
    



    # 🟩 Now begin loading content vocabulary
    vocab_CC_frequent_ints = list(range(-MAX_INT_TOKEN, MAX_INT_TOKEN))

    vocab_CC_digit_char = ['0','1','2','3','4','5','6','7','8','9']
    vocab_CC_normal_char = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    vocab_CC_normal_char += [s.upper() for s in vocab_CC_normal_char]
    from string import punctuation
    unknown_type_token = '<<||unknown_type_token||>>'
    too_tiny_float_token = '<<||too_tiny_float_token||>>'
    too_big_float_token = '<<||too_big_float_token||>>'

    reserved_tokens = [f'<<||SPECIAL_RESERVED_TOKEN_POSSIBLY_NEVER_USED_{i}||>>' for i in range(20)]  # reserved_tokens[7] is for unseen syntax. Other tokens are currently not used, and may never be used.

    vocab_CC_other_char = ['-', '_', '(', ')', '[', ']', '{', '}', '.', '*', '<', '>', '=', '==', ' ', '\n', '\t', '\r', '', "$", "!", "#", "%", "^", "&", "*", "@", "+", "?", "'", "/"] + list(punctuation) + [unknown_type_token, too_tiny_float_token, too_big_float_token] + reserved_tokens + [b'\\n', b'-', b'+', b'', b'0', b'1', b'\n', b' ']

    vocab_CC_other_char = list(set(vocab_CC_other_char))

    VOCAB_CC_float = [0.0, 0.00001, 0.0001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0, 1e-09, 1e-08, 1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    # (np.arange(101)/100).tolist() + (np.arange(101)/10).tolist() + [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1E6, 1e7, 1E8, 1e9]
    


    solution_func_name = 'syntax_transformer_wrap_func'
    vocab_CC_diy_var_name = [f'var_{i}' for i in range(MAX_DIY_NAME_TOKEN)]
    vocab_CC_diy_func_name = [f'func_{i}' for i in range(MAX_DIY_NAME_TOKEN)]
    vocab_CC_diy_class_name = [f'Class_{i}' for i in range(MAX_DIY_NAME_TOKEN)]
    vocab_CC_diy_input_arg_name = [f'var_in_{i}' for i in range(MAX_DIY_NAME_TOKEN)]
    vocab_CC_name_pool = [solution_func_name] + vocab_CC_diy_var_name + vocab_CC_diy_func_name + vocab_CC_diy_class_name + vocab_CC_diy_input_arg_name


    def get_built_in_basic(): # run this and fix; otherwise the result may be different on different machine; vocabulary MUST be a fixed one.
        vocab_CC_built_in_string = \
            list(__builtins__.keys()) + \
            list(dir('s')) + \
            list(dir(1)) + \
            list(dir(float(0.1))) + \
            list(dir(None)) + \
            list(dir(True)) + \
            list(dir(dict())) + \
            list(dir(set())) + \
            list(dir([]))
        return vocab_CC_built_in_string

    vocab_CC_built_in_string = ['__name__', '__doc__', '__package__', '__loader__', '__spec__', '__build_class__', '__import__', 'abs', 'all', 'any', 'ascii', 'bin', 'breakpoint', 'callable', 'chr', 'compile', 'delattr', 'dir', 'divmod', 'eval', 'exec', 'format', 'getattr', 'globals', 'hasattr', 'hash', 'hex', 'id', 'input', 'isinstance', 'issubclass', 'iter', 'len', 'locals', 'max', 'min', 'next', 'oct', 'ord', 'pow', 'print', 'repr', 'round', 'setattr', 'sorted', 'sum', 'vars', 'None', 'Ellipsis', 'NotImplemented', 'False', 'True', 'bool', 'memoryview', 'bytearray', 'bytes', 'classmethod', 'complex', 'dict', 'enumerate', 'filter', 'float', 'frozenset', 'property', 'int', 'list', 'map', 'object', 'range', 'reversed', 'set', 'slice', 'staticmethod', 'str', 'super', 'tuple', 'type', 'zip', '__debug__', 'BaseException', 'Exception', 'TypeError', 'StopAsyncIteration', 'StopIteration', 'GeneratorExit', 'SystemExit', 'KeyboardInterrupt', 'ImportError', 'ModuleNotFoundError', 'OSError', 'EnvironmentError', 'IOError', 'EOFError', 'RuntimeError', 'RecursionError', 'NotImplementedError', 'NameError', 'UnboundLocalError', 'AttributeError', 'SyntaxError', 'IndentationError', 'TabError', 'LookupError', 'IndexError', 'KeyError', 'ValueError', 'UnicodeError', 'UnicodeEncodeError', 'UnicodeDecodeError', 'UnicodeTranslateError', 'AssertionError', 'ArithmeticError', 'FloatingPointError', 'OverflowError', 'ZeroDivisionError', 'SystemError', 'ReferenceError', 'MemoryError', 'BufferError', 'Warning', 'UserWarning', 'DeprecationWarning', 'PendingDeprecationWarning', 'SyntaxWarning', 'RuntimeWarning', 'FutureWarning', 'ImportWarning', 'UnicodeWarning', 'BytesWarning', 'ResourceWarning', 'ConnectionError', 'BlockingIOError', 'BrokenPipeError', 'ChildProcessError', 'ConnectionAbortedError', 'ConnectionRefusedError', 'ConnectionResetError', 'FileExistsError', 'FileNotFoundError', 'IsADirectoryError', 'NotADirectoryError', 'InterruptedError', 'PermissionError', 'ProcessLookupError', 'TimeoutError', 'open', 'quit', 'exit', 'copyright', 'credits', 'license', 'help', 'execfile', 'runfile', '__pybind11_internals_v4_gcc_libstdcpp_cxxabi1011__', '__add__', '__class__', '__contains__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__getnewargs__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__iter__', '__le__', '__len__', '__lt__', '__mod__', '__mul__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__rmod__', '__rmul__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 'capitalize', 'casefold', 'center', 'count', 'encode', 'endswith', 'expandtabs', 'find', 'format', 'format_map', 'index', 'isalnum', 'isalpha', 'isascii', 'isdecimal', 'isdigit', 'isidentifier', 'islower', 'isnumeric', 'isprintable', 'isspace', 'istitle', 'isupper', 'join', 'ljust', 'lower', 'lstrip', 'maketrans', 'partition', 'replace', 'rfind', 'rindex', 'rjust', 'rpartition', 'rsplit', 'rstrip', 'split', 'splitlines', 'startswith', 'strip', 'swapcase', 'title', 'translate', 'upper', 'zfill', '__abs__', '__add__', '__and__', '__bool__', '__ceil__', '__class__', '__delattr__', '__dir__', '__divmod__', '__doc__', '__eq__', '__float__', '__floor__', '__floordiv__', '__format__', '__ge__', '__getattribute__', '__getnewargs__', '__gt__', '__hash__', '__index__', '__init__', '__init_subclass__', '__int__', '__invert__', '__le__', '__lshift__', '__lt__', '__mod__', '__mul__', '__ne__', '__neg__', '__new__', '__or__', '__pos__', '__pow__', '__radd__', '__rand__', '__rdivmod__', '__reduce__', '__reduce_ex__', '__repr__', '__rfloordiv__', '__rlshift__', '__rmod__', '__rmul__', '__ror__', '__round__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__', '__rtruediv__', '__rxor__', '__setattr__', '__sizeof__', '__str__', '__sub__', '__subclasshook__', '__truediv__', '__trunc__', '__xor__', 'as_integer_ratio', 'bit_length', 'conjugate', 'denominator', 'from_bytes', 'imag', 'numerator', 'real', 'to_bytes', '__abs__', '__add__', '__bool__', '__class__', '__delattr__', '__dir__', '__divmod__', '__doc__', '__eq__', '__float__', '__floordiv__', '__format__', '__ge__', '__getattribute__', '__getformat__', '__getnewargs__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__int__', '__le__', '__lt__', '__mod__', '__mul__', '__ne__', '__neg__', '__new__', '__pos__', '__pow__', '__radd__', '__rdivmod__', '__reduce__', '__reduce_ex__', '__repr__', '__rfloordiv__', '__rmod__', '__rmul__', '__round__', '__rpow__', '__rsub__', '__rtruediv__', '__set_format__', '__setattr__', '__sizeof__', '__str__', '__sub__', '__subclasshook__', '__truediv__', '__trunc__', 'as_integer_ratio', 'conjugate', 'fromhex', 'hex', 'imag', 'is_integer', 'real', '__bool__', '__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__abs__', '__add__', '__and__', '__bool__', '__ceil__', '__class__', '__delattr__', '__dir__', '__divmod__', '__doc__', '__eq__', '__float__', '__floor__', '__floordiv__', '__format__', '__ge__', '__getattribute__', '__getnewargs__', '__gt__', '__hash__', '__index__', '__init__', '__init_subclass__', '__int__', '__invert__', '__le__', '__lshift__', '__lt__', '__mod__', '__mul__', '__ne__', '__neg__', '__new__', '__or__', '__pos__', '__pow__', '__radd__', '__rand__', '__rdivmod__', '__reduce__', '__reduce_ex__', '__repr__', '__rfloordiv__', '__rlshift__', '__rmod__', '__rmul__', '__ror__', '__round__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__', '__rtruediv__', '__rxor__', '__setattr__', '__sizeof__', '__str__', '__sub__', '__subclasshook__', '__truediv__', '__trunc__', '__xor__', 'as_integer_ratio', 'bit_length', 'conjugate', 'denominator', 'from_bytes', 'imag', 'numerator', 'real', 'to_bytes', '__class__', '__contains__', '__delattr__', '__delitem__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__iter__', '__le__', '__len__', '__lt__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__setattr__', '__setitem__', '__sizeof__', '__str__', '__subclasshook__', 'clear', 'copy', 'fromkeys', 'get', 'items', 'keys', 'pop', 'popitem', 'setdefault', 'update', 'values', '__and__', '__class__', '__contains__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__iand__', '__init__', '__init_subclass__', '__ior__', '__isub__', '__iter__', '__ixor__', '__le__', '__len__', '__lt__', '__ne__', '__new__', '__or__', '__rand__', '__reduce__', '__reduce_ex__', '__repr__', '__ror__', '__rsub__', '__rxor__', '__setattr__', '__sizeof__', '__str__', '__sub__', '__subclasshook__', '__xor__', 'add', 'clear', 'copy', 'difference', 'difference_update', 'discard', 'intersection', 'intersection_update', 'isdisjoint', 'issubset', 'issuperset', 'pop', 'remove', 'symmetric_difference', 'symmetric_difference_update', 'union', 'update', '__add__', '__class__', '__contains__', '__delattr__', '__delitem__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__iadd__', '__imul__', '__init__', '__init_subclass__', '__iter__', '__le__', '__len__', '__lt__', '__mul__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__rmul__', '__setattr__', '__setitem__', '__sizeof__', '__str__', '__subclasshook__', 'append', 'clear', 'copy', 'count', 'extend', 'index', 'insert', 'pop', 'remove', 'reverse', 'sort']




        
    vocab_CC_built_in_non_string = [None, True, False]

    content_final_chasing_syntax_token = '<<||content_final_chasing_syntax_token||>>'
    syntax_on_hold_for_content_token = '<<||syntax_on_hold_for_content_token||>>'

    input_cross_sample_pad_syn_token = '<<||input_cross_sample_pad_syn_token||>>'
    input_cross_sample_pad_cont_token = '<<||input_cross_sample_pad_cont_token||>>'
    cross_instance_pad_syn_token = '<<||cross_instance_pad_syn_token||>>'
    cross_instance_pad_cont_token = '<<||cross_instance_pad_cont_token||>>'
    bos_token = '<<||bos_token||>>'
    eos_token = '<<||eos_token||>>'
    s1234_sep_token = reserved_tokens[0]




    system_SI = [
        syntax_on_hold_for_content_token,
        input_cross_sample_pad_syn_token,
        cross_instance_pad_syn_token,
        ]
    system_SO = [
        bos_token, eos_token,
        syntax_on_hold_for_content_token,
        cross_instance_pad_syn_token,
        ]
    system_CC = [
        content_final_chasing_syntax_token,
        cross_instance_pad_cont_token,
        input_cross_sample_pad_cont_token,
        ]






    # 🟩 Now begin summarizing vocabulary: 
    _vocab_all_SI_list = \
        system_SI + \
        vocab_SI_saved + \
        vocab_SO_NLP_syntax_markings
        
    _vocab_all_SO_list = \
        system_SO + \
        vocab_SO_saved

    _vocab_all_CC_list = \
        system_CC + \
        vocab_CC_frequent_ints + \
        vocab_CC_digit_char + \
        vocab_CC_normal_char + \
        vocab_CC_other_char + \
        VOCAB_CC_float + \
        vocab_CC_built_in_non_string + \
        vocab_CC_built_in_string + \
        vocab_CC_name_pool + \
        vocab_CC_wild


    # 🟩 Now deal with the buggist thing...
    _vocab_CC_syntax_class = sorted(get_all_definitions(python_syntax).values(), key=lambda x: repr(x))
    _vocab_CC_syntax_class = list(filter(lambda x: hasattr(x.__init__, '__code__') and x.__init__.__code__.co_argcount<=1, _vocab_CC_syntax_class))
    syntax_cls_token_num = len(_vocab_CC_syntax_class)
    assert syntax_cls_token_num == 35







    class Vocabulary_Defs:
        """ The core vocabulary class that handles token -> int, int -> token convert, and manage the covabulary.
        """

        def __init__(self):
            self.refuse_unseen_tokens = REFUSE_UNSEEN_TOKENS
            self.VOCAB_SO_FILE = VOCAB_SO_FILE
            self.VOCAB_SI_FILE = VOCAB_SI_FILE
            self.VOCAB_CC_FILE = VOCAB_CC_FILE

            # 🟩 constant tokens related to the great wall
            self.initial_vi_set = set(_vocab_all_SI_list + _vocab_all_CC_list)
            self.initial_vo_set = set(_vocab_all_SO_list + _vocab_all_CC_list)
            self.unseen_token = reserved_tokens[7]

            # 🟩 load the great wall carefully
            self.reset_the_great_wall_as(VOCAB_GREAT_WALL_FILE)


            # 🟩 constant groups
            self.vocab_CC_wild = vocab_CC_wild
            self.vocab_SO_saved = vocab_SO_saved
            self.vocab_SI_saved = vocab_SI_saved


            # 🟩 constant tokens
            self.vocab_CC_built_in_string = vocab_CC_built_in_string
            self.input_cross_sample_pad_syn_token = input_cross_sample_pad_syn_token
            self.input_cross_sample_pad_cont_token = input_cross_sample_pad_cont_token
            self.cross_instance_pad_syn_token = cross_instance_pad_syn_token
            self.cross_instance_pad_cont_token = cross_instance_pad_cont_token
            self.s1234_sep_token = s1234_sep_token

            self.syntax_on_hold_for_content_token = syntax_on_hold_for_content_token
            self.content_final_chasing_syntax_token = content_final_chasing_syntax_token


            # 🟩 constant special name tokens
            self.solution_func_name = solution_func_name
            self.vocab_CC_diy_class_name = vocab_CC_diy_class_name
            self.vocab_CC_diy_func_name = vocab_CC_diy_func_name
            self.vocab_CC_diy_input_arg_name = vocab_CC_diy_input_arg_name
            self.vocab_CC_diy_var_name = vocab_CC_diy_var_name
            self._vocab_CC_syntax_class = _vocab_CC_syntax_class
            self.VOCAB_CC_float = VOCAB_CC_float
            self.unknown_type_token = unknown_type_token

            return



        def reset_the_great_wall_as(self, VOCAB_GREAT_WALL_FILE):
            self.VOCAB_GREAT_WALL_FILE = VOCAB_GREAT_WALL_FILE

            if not os.path.exists(self.VOCAB_GREAT_WALL_FILE):
                self._voca_list_I = _vocab_CC_syntax_class +  sorted(self.initial_vi_set, key = lambda x: repr(x))
                self._voca_list_O = _vocab_CC_syntax_class +  sorted(self.initial_vo_set, key = lambda x: repr(x))
                self.save_the_great_wall()
                print(f' ❗️❗️ Previous Static vocabs ("the great wall") does not exist ❗️❗️ You should ONLY see this message ONCE!!! ')
                
            self.load_the_great_wall()
            return

        def save_the_great_wall(self):
            """ The syntax cls token cannot be saved, otherwise when load and eval it, will syntax error.
            THese tokens are at the beginning.
            mode: 'r' or 'w'
            """

            vi = self._voca_list_I[syntax_cls_token_num:]
            vo = self._voca_list_O[syntax_cls_token_num:]

            print([vi, vo], file = open(self.VOCAB_GREAT_WALL_FILE, 'w'))
            return self.VOCAB_GREAT_WALL_FILE

        def load_the_great_wall(self):
            """ Ensure:
                1. initial is contained in loaded voca
                2. insect syntax class to the front
                3. reload everything related to the great wall
            """
            _voca_list_I, _voca_list_O = eval(load_txt(self.VOCAB_GREAT_WALL_FILE))

            self._voca_list_I = _vocab_CC_syntax_class + _voca_list_I
            self._voca_list_O = _vocab_CC_syntax_class + _voca_list_O

            # 🟩 all vocabs
            self.set_vocabs(self._voca_list_I, self._voca_list_O)
            self.init_padding_token_ids()

            return

        
        def is_unseen_SI(self, tok):
            return tok not in self.t2i_inp
        def is_unseen_SO(self, tok):
            return tok not in self.t2i_out
        def is_unseen_CC(self, tok):
            key = self.toKey(tok, 'I', 'cont', need_update=False)
            return key not in self.t2i_inp



        def update(self, newtokens_list=[], I_or_O_list=[], syn_or_cont_list=[]): # types can be: SI/SO/CI/CO
            if self.refuse_unseen_tokens:
                raise ValueError(f'Unseen Tokens: {newtokens_list}')

            for t, I_or_O, syn_or_cont in zip(newtokens_list, I_or_O_list, syn_or_cont_list):
                if I_or_O=='I':
                    assert t not in self.t2i_inp
                    self._voca_list_I.append(t)
                    if syn_or_cont=='cont':
                        self._voca_list_O.append(t)
                elif I_or_O=='O':
                    assert t not in self.t2i_out
                    self._voca_list_O.append(t)
                    if syn_or_cont=='cont':
                        self._voca_list_I.append(t)
                        


            # re-init the vocabulary
            self.set_vocabs(self._voca_list_I, self._voca_list_O)

            return

        def toKey(self, token, I_or_O, syn_or_cont, need_update=True):
            if obj_is_syntax_class(token):
                key = type(token)
            else:
                key = token
            if not need_update:
                return key
            if (I_or_O=='I') and (key not in self.t2i_inp) or \
               (I_or_O=='O') and (key not in self.t2i_out):
                if robust_tokenization_after_fix_vocab:
                    return self.unseen_token
                self.update([key], [I_or_O], [syn_or_cont])
            return key

        def toToken(self, key):
            if cls_is_syntax_class(key):
                token = key()
            else:
                token = key
            return token

        def token2int_I(self, token, syn_or_cont):
            """ Designed for the transformer Input """
            key = self.toKey(token, 'I', syn_or_cont)
            idx = self.t2i_inp[key]
            return idx
        def int2token_I(self, idx):
            """ Designed for the transformer Input """
            token = self.i2t_inp[idx]
            token = self.toToken(token)
            return token
        def token2int_O(self, token, syn_or_cont):
            """ Designed for the transformer Output, union dict """
            key = self.toKey(token, 'O', syn_or_cont)
            idx = self.t2i_out[key]
            return idx
        def int2token_O(self, idx):
            """ Designed for the transformer Output, union dict """
            token = self.i2t_out[idx]
            token = self.toToken(token)
            return token
            
        def set_vocabs(self, vi,vo):
            self._voca_list_I, self._voca_list_O = vi,vo

            self.t2i_inp,  self.i2t_inp,    \
            self.t2i_out,  self.i2t_out     =\
                convert_lists_to_those_dicts(self._voca_list_I, self._voca_list_O)
            return
        def show_vocab(self):
            print(f'Input, Output vocab size = {len(self._voca_list_I)} , {len(self._voca_list_O)}')
            return

        def init_padding_token_ids(self):
            self.output_pad_id = self.t2i_out[cross_instance_pad_syn_token]

            self.input_cross_instance_pad_syn_id = self.t2i_inp[cross_instance_pad_syn_token]
            self.input_cross_instance_pad_cont_id = self.t2i_inp[cross_instance_pad_cont_token]
            self.input_cross_sample_pad_syn_id = self.t2i_inp[input_cross_sample_pad_syn_token]
            self.input_cross_sample_pad_cont_id = self.t2i_inp[input_cross_sample_pad_cont_token]


            self.s1234_sep_id = self.t2i_out[s1234_sep_token]
            self.bos_token_id = self.t2i_out[bos_token]
            self.eos_token_id = self.t2i_out[eos_token]
            self.distill_NLP_syntax_marking_ids = [self.t2i_inp[x] for x in vocab_SO_NLP_syntax_markings]


    vocabulary_defs = Vocabulary_Defs()
    return vocabulary_defs

vocabulary_defs = get_vocabulary_defs()











def simple_dfs_template(root):
    """
    Empty Algorithm template for compressed serialization of syntax tree - it works.
    """
    root = copy.deepcopy(root)
    
    visit = lambda node: print((node, node.__class__.__name__))
    def visit_demo2(x):
        if x.__class__.__name__=='Constant':
            print(x.value)
            print(x.content_string)
            print


    visit(root)
    stack = [root]
    while stack:

        while stack and stack[-1].children==[]:
            stack.pop()
        if not stack:
            break
        
        while stack[-1].children:
            
            father = stack[-1]
            cur_node = father.children.pop(0)
            
            visit(cur_node) # visit node just before pushing in.
            
            stack.append(cur_node)

    return 



def convert_tree_obj_to_token_sequence(root, max_compress=1,is_debugging=0):
    """
    Algorithm for compressed serialization of syntax tree.
    """
    stack = [root]
    syntax_sequence = [root.primary_string]
    content_sequence = [root.content_string]
    while stack:
        # do the wrap-up work for previously visited nodes
        cur_syntax = ''
        while stack and stack[-1].children==[]:
            cur_syntax += stack.pop().closing_string
        if not stack:
            break
        
        # now begin a new round of grouping syntax nodes
        split_cnt = 0
        
        # now start stepping down
        # stop grouping case 2/2: no children available
        while split_cnt<=1 and stack[-1].children:
            # make sure split_cnt==1 only encounter once
            if split_cnt>0: 
                split_cnt += 1
            
            father = stack[-1]
            cur_node = father.children.pop(0)
            glue_pre, glue_post = father.children_glue_strings.pop(0)
            cur_syntax += glue_pre
            cur_syntax += cur_node.primary_string
            cur_node.closing_string += glue_post
            stack.append(cur_node)

            # stop grouping case 1/2: made non-trivial children selection only once. 
            if len(cur_node.children)>1:
                split_cnt += 1
        cur_content = cur_node.content_string

        syntax_sequence.append(cur_syntax)
        if is_debugging:
            print(f'{cur_content}\t\t{cur_syntax}')
        cur_syntax = ''
        content_sequence.append(cur_content)
    syntax_sequence.append(cur_syntax)

    assert len(syntax_sequence) == len(content_sequence)+1
    assert syntax_sequence[-1]

    if not max_compress:
        return syntax_sequence, content_sequence
    else:
        compresssed_syn, compressed_cont = [], []
        s_holder = []
        for s,c in zip(syntax_sequence, content_sequence):
            s_holder.append(s)
            if c:
                compresssed_syn.append(''.join(s_holder))
                compressed_cont.append(c)
                s_holder = []

        compresssed_syn.append(''.join(s_holder)+syntax_sequence[-1])
        return compresssed_syn, compressed_cont



global_dict = get_all_definitions(python_syntax)
global_dict.update({
    'convert_tree_obj_to_token_sequence': convert_tree_obj_to_token_sequence,
    'astunparse': astunparse
    })
assert len(global_dict)>=96, len(global_dict)




def get_exec_prints(code_block_str):
    import sys
    from io import StringIO
    import contextlib

    @contextlib.contextmanager
    def stdoutIO(stdout=None):
        old = sys.stdout
        if stdout is None:
            stdout = StringIO()
        sys.stdout = stdout
        yield stdout
        sys.stdout = old
    
    with stdoutIO() as s:
        print(code_block_str,file=open('_log_error.py','w'))
        exec(code_block_str, global_dict)  # Here it might report error: 'name xxx is not defined'. This is due to incomplete implementation in python_syntax.py. To fix: print out the syntax tree, and find that name 'xxx', then go into python_syntax.py, choose any existing class as a template, implement the class 'xxx', then done.


    printed = s.getvalue()
    return printed





def tonp(arr):
    try:
        import torch
        is_torch = type(arr) is torch.Tensor
    except ImportError:
        is_torch = False
    if is_torch:
        return arr.detach().cpu().data.numpy()
    elif hasattr(arr, 'numpy'): # <class 'tensorflow.python.framework.ops.EagerTensor'>
        return arr.numpy()
    elif hasattr(arr, 'result'): # tf2.keras.metrics.base_metric.Mean
        return arr.result().numpy()
    else:
        return np.asarray(arr)




def step1_convert_code_str_to_entire_synTree_str(python_code_str, indent=4, check_runable = False):

    if check_runable:
        exec(python_code_str)
    tree = ast.parse(python_code_str)

    # below equal to:   entire_tree_str = astunparse.dump(tree)
    v = cStringIO()
    astunparse.Printer(file=v, indent=" "*indent).visit(tree)
    entire_tree_str = v.getvalue()

    return entire_tree_str


def step2_convert_entire_synTree_str_to_tokenized_sequence(entire_tree_str):

    cmd_newTree = 'tree = ' + entire_tree_str
    cmd_tokenize = 'syntax_sequence, content_sequence = convert_tree_obj_to_token_sequence(tree)'
    cmd_print = 'print(syntax_sequence)\nprint(content_sequence)'
    cmd_to_run = '\n\n'.join([cmd_newTree, cmd_tokenize, cmd_print])

    outputs = get_exec_prints(cmd_to_run)
    syntax_sequence, content_sequence, _ = outputs.split('\n')

    syntax_sequence = eval(syntax_sequence)
    content_sequence = eval(content_sequence)

    assert len(syntax_sequence) == len(content_sequence) + 1
    return syntax_sequence, content_sequence


def step3_glue_context_content_seqs(syntax_sequence, content_sequence):
    assert len(syntax_sequence) == len(content_sequence)+1
    code = [syntax_sequence.pop(0)]
    for syn, cont in zip(syntax_sequence, content_sequence):
        code.append(cont)
        code.append(syn)
    entire_synTree_str = ''.join(code)
    return entire_synTree_str

def step4_convert_entire_synTree_str_to_code_str(synTree_str):
    
    cmd_newTree = 'synTree_obj = ' + synTree_str
    cmd_convert_to_code = 'str_formed_code = astunparse.unparse(synTree_obj)'
    cmd_print = 'print(str_formed_code)'
    
    cmd_to_run = '\n\n'.join([cmd_newTree, cmd_convert_to_code, cmd_print])
    recov_code_str = get_exec_prints(cmd_to_run)

    return recov_code_str

str2obj = lambda code_str: eval(step1_convert_code_str_to_entire_synTree_str(code_str))


def sample_encoder(data_sample, no_content_tok = '<NO_CONTENT>'):
    """ Transform one sample into encoded token sequence.

    Args:
        data_sample: a sample is a 1-D list of input or outputs, which could be a mixture of int, float, string, list, or whatever.
            form:
                [var1, ..., var_N]
    Returns:
        encoded: a list of encoded; length not fiex (not yet padded).
            form:
                [(syntax1, content1), (syn2, cont2), ..., (syn_N, NO_CONTENT)]
    """

    data_sample = repr(data_sample)
    synTree_str = step1_convert_code_str_to_entire_synTree_str(data_sample, indent=2, check_runable = False)
    syntax_sequence, content_sequence = step2_convert_entire_synTree_str_to_tokenized_sequence(synTree_str)
    content_sequence.append(no_content_tok)
    encoded = [(cont, syn) for cont, syn in zip(syntax_sequence, content_sequence)]
    return encoded


def replace_elems_in_a_list(tar_list, locs, tar_value):
    for loc in locs:
        tar_list[loc] = tar_value
    return tar_list


def python_repr_tokenizer(python_repr, is_iodata, coarse_level=False):
    """ string -> string tokens
    Tokenizing the code/iodata into syntax and content tokens. 
    The content tokens are NOT evaluated into python obj for better transferrability.
    All vocabulary classification:
    - non-string tokens: int, a few DIY syntax classes (USub()), None, True, False
    - string tokens: all syntax tokens, built-in functions, chars
    - classification:
        - non_lios: non long int or string - directly return single token
        - two cases need carefulness: string and long int in content tokens.
            - long_int
            - types of content string:
                - built_in_func_name: keep it unchanged.
                - top_level_answer_func_name: replace it with the single fixed answer name (done elsewhere).
                * user_defined_func_name:   replace from pool
                * user_defined_var_name:    replace from pool
                * user_defined_class_name:  replace from pool
                - string_value: always break it down to letter chars/string digits/other chars one by one.
    """

    # 🟩 slower, import name replace correct
    import_linked_names, old_syn_seq, old_cont_seq = remove_doc_str_and_detect_import_linked_names_acc_v3(python_repr)

    old_cont_seq = [eval(x) for x in old_cont_seq]
    assert len(old_syn_seq) == len(old_cont_seq) + 1

    diy_names = defaultdict(list)
    syn_seq, cont_seq = [], []

    # 🟩 reduce the name space by de-generating names (replace with name pool elements), and deal with too long tokens.
    for syn, cont in zip(old_syn_seq, old_cont_seq):
        
        if type(cont) is int:  # 🟩 break down long ints
            if cont>MAX_INT_TOKEN: # 'long_int', break it down
                cont_list = [int(c) for c in str(cont)]
                syn_list = [syn] + [vocabulary_defs.syntax_on_hold_for_content_token]*(len(cont_list)-1)
                syn_seq.extend(syn_list); cont_seq.extend(cont_list)
            else:
                syn_seq.append(syn); cont_seq.append(cont)
        elif type(cont) is str:  # 🟩 break down string with len > 1
            if syn.endswith('ClassDef(name='):
                appear_as = 'is_classdef'
            elif syn.endswith('FunctionDef(name='):
                appear_as = 'is_funcdef'
            elif syn.endswith('arg='):
                appear_as = 'is_argdef'
            elif syn.endswith('func=Name(id='):
                appear_as = 'is_funccall'   # could be func/class/obj
            elif syn.endswith('Name(id='):
                appear_as = 'is_name'  # undefinite now, could be var/func/class/obj
            elif syn.endswith('),name='):
                appear_as = 'is_othername'  # undefinite now, could be var/func/class/obj
            elif syn.endswith('Constant(value='):
                appear_as = 'is_constvalue'  # break down
            elif syn.endswith('attr='):
                appear_as = 'is_attr'
            elif syn.endswith('alias(name='):
                appear_as = 'is_alias'
            elif syn.endswith('ImportFrom(module='):
                appear_as = 'is_import_from'
            elif syn.endswith('asname='):
                appear_as = 'is_asname'
            elif syn.endswith('Global(names=['):
                appear_as = 'is_Global'
            elif syn.endswith('Nonlocal(names=['):
                appear_as = 'is_Nonlocal'
            elif syn==',':
                appear_as = 'is_comma'
            else:
                if VERBOSE:
                    print(f'During tokenizer for string, when detecting string type, implementation seems incomplete, newly found type is: {syn}')
                appear_as = 'is_unknown'

            if appear_as == 'is_constvalue':  # 🟩 'string_value', break it down
                if len(cont)<=1:
                    
                    syn_seq.append(syn); cont_seq.append(cont)
                else:
                    cont_list = list(cont)
                    syn_list = [syn] + [vocabulary_defs.syntax_on_hold_for_content_token]*(len(cont_list)-1)
                    syn_seq.extend(syn_list); cont_seq.extend(cont_list)
            else:

                # 🟩 replace 'diy name' with name pool
                keep_untouched = False
                if is_iodata:
                    keep_untouched = True
                elif cont in import_linked_names:
                    keep_untouched = True
                elif cont == vocabulary_defs.solution_func_name:
                    keep_untouched = True


                if keep_untouched:
                    syn_seq.append(syn); cont_seq.append(cont)
                else:  # 🟩 it's time to replace it with name pool! Mark them first, replace all later.
                    if cont not in diy_names:
                        diy_names[cont] = ({
                                'locs': [],
                                'appear_as': set()
                        })
                    diy_names[cont]['locs'].append(len(cont_seq))
                    diy_names[cont]['appear_as'].update([appear_as])
                    syn_seq.append(syn); cont_seq.append(cont)

        elif type(cont) is float:  # 🟩 try convert to nearest
            
            find_closest_float_token = lambda cont: closest(cont, vocabulary_defs.VOCAB_CC_float)
            cont = find_closest_float_token(cont)
            syn_seq.append(syn); cont_seq.append(cont)

        else:
            if vocabulary_defs.is_unseen_CC(cont):
                if is_iodata:
                    cont = vocabulary_defs.unknown_type_token   # Logic is: if transformer input unseen token, accept it, replace with a special token; if output unseen token, refuse it.
                    syn_seq.append(syn); cont_seq.append(cont)
                elif type(cont) is bytes:
                    syn_seq.append(syn); cont_seq.append(cont)
                    print(f'Bytes token found: {type(cont)},  {cont}')

                else:
                    raise ValueError(f'Token type not supported: {type(cont)},  {cont}  , refuse this sample.')
            else:
                syn_seq.append(syn); cont_seq.append(cont)

    # 🟩 Now execute the name replacement.
    diyClass_name_to_replace = []
    diyFunc_name_to_replace = []
    inputArg_name_to_replace = []
    diyVar_name_to_replace = []
    for name in diy_names:
        if  'is_classdef' in diy_names[name]['appear_as']:
            diyClass_name_to_replace.append(name)
        elif  'is_funcdef' in diy_names[name]['appear_as']:
            diyFunc_name_to_replace.append(name)
        elif 'is_argdef' in diy_names[name]['appear_as']:
            inputArg_name_to_replace.append(name)
        else:
            diyVar_name_to_replace.append(name)

    def batch_replace_all_diy_name(cont_seq, existing_names, tar_names):
        if len(existing_names)>len(tar_names):
            raise UserWarning('❗️ ❗️ Tokenization fail, too many user-defined variables; this should rarely happen.')
        for src_name, tar_name in zip(existing_names, tar_names[:len(existing_names)]):
            locs = diy_names[src_name]['locs']
            cont_seq = replace_elems_in_a_list(cont_seq, locs, tar_name)
        return cont_seq
    try:
        cont_seq = batch_replace_all_diy_name(cont_seq, diyClass_name_to_replace, vocabulary_defs.vocab_CC_diy_class_name)
        cont_seq = batch_replace_all_diy_name(cont_seq, diyFunc_name_to_replace, vocabulary_defs.vocab_CC_diy_func_name)
        cont_seq = batch_replace_all_diy_name(cont_seq, inputArg_name_to_replace, vocabulary_defs.vocab_CC_diy_input_arg_name)
        cont_seq = batch_replace_all_diy_name(cont_seq, diyVar_name_to_replace, vocabulary_defs.vocab_CC_diy_var_name)
    except:
        return [], []

    # 🟩 Name replacement all finished, do some final conversions.

    syn_seq.append(old_syn_seq[-1])
    assert len(syn_seq) == len(cont_seq) + 1
    coarse = lambda lst: (lst[:1] if len(lst) >= 1 else lst) + [lst[i] for i in range(len(lst)) if (i+1) % 500 == 0]
    if coarse_level:
        syn_seq.insert(0, coarse(syn_seq))
        cont_seq.insert(0, coarse(cont_seq))
    return syn_seq, cont_seq


def print_tokens(syns, conts):
    print('🟧' + '-'*10 + '🟧')
    for cur_syntax, cur_content in zip(syns, conts):
        print(f'{repr(cur_content)}\t\t{repr(cur_syntax)}')
    if len(syns)>len(conts):
        print(syns[len(conts):])
    print('🟧' + '-'*10 + '🟧')


def closest(v, grid):
    """ Find the nearest point in the grid that is close to v. Data types of grid and v can be different; always return grid value.
    """
    from heapq import nsmallest
    cont = nsmallest(1, list(grid), key=lambda x: abs(x-v))[0]
    return cont


def remove_doc_str_and_detect_import_linked_names_acc_v3(python_repr):

    synTree_str = step1_convert_code_str_to_entire_synTree_str(python_repr, indent=4, check_runable = False)

    root = eval(synTree_str) # If your python env is older than 3.8 (or, significantly newer), this line may report error "XXX is not defined." To fix, you'll have to define it in tokenizer/python_syntax.py, follow the parsed AST (synTree_str) of your python version. Current implementations in python_syntax.py only supports python 3.8 at the time of implementation.

    import_linked_names = set(vocabulary_defs.vocab_CC_built_in_string)

    def virtual_child_empty(father):
        if not hasattr(father, 'poped_children_num'):
            return father.children==[]  # or 
        else:
            return father.poped_children_num==len(father.children)
    def virtual_pop_child(father):
        if not hasattr(father, 'poped_children_num'):
            father.poped_children_num = 0
        father.poped_children_num += 1
        return father.children[father.poped_children_num-1]

    def is_doc_str(x):
        if x.__class__.__name__=='Expr' and x.value.__class__.__name__=='Constant':# this is a doc string!!! remove it!!!
            return Pass(), True
        else: 
            return x, False

    def find_import_linked_names(x):
        linked_names = []
        if x.__class__.__name__ in ['ImportFrom', 'Import']:
            this = list(map(lambda y: y.name, x.names))
            this2 = list(itertools.chain.from_iterable([x.split('.') for x in this]))
            linked_names = linked_names + this + this2
            print
        if x.__class__.__name__ == 'ImportFrom':
            linked_names += [x.module]
            linked_names += x.module.split('.')
            print
        return linked_names
    SCB_types = set([
        'ImportFrom', 'Import', 
        'FunctionDef', 'ClassDef', 
        'Assign', 'AugAssign', 'AnnAssign', 
        'Expr', 'If', 'IfExp', 'GeneratorExp', 'NamedExpr', 
        'Global', 'Nonlocal'
        'While', 'For', 'Pass', 'Continue', 
        'Try', 'ExceptHandler', 'Raise', 'Assert', 'Delete', 'With',
        'Return', 'Yield', 'YieldFrom', 
        ])
    def collect_self_complete_block_info(x):
        return (x.__class__.__name__ in SCB_types), x.__class__.__name__, x.content_string, x.primary_string

    scb_visit_list = []
    stack = [root]

    syntax_sequence = [root.primary_string]
    content_sequence = [root.content_string]

    while stack:
        # do the wrap-up work for previously visited nodes
        cur_syntax = ''

        while stack and virtual_child_empty(stack[-1]):
            # stack.pop()
            cur_syntax += stack.pop().closing_string
        if not stack:
            
            break
        
        # now begin a new round of grouping syntax nodes
        split_cnt = 0
        
        while split_cnt<=1 and (not virtual_child_empty(stack[-1])):
            # make sure split_cnt==1 only encounter once
            if split_cnt>0: 
                split_cnt += 1
            
            father = stack[-1]
            cur_node = virtual_pop_child(father)

            glue_pre, glue_post = father.children_glue_strings.pop(0)
            cur_syntax += glue_pre
            cur_syntax += cur_node.primary_string
            cur_node.closing_string += glue_post

            # 🟩 visit node: remove doc string
            _cur_node, is_dec_str = is_doc_str(cur_node) # visit node just before pushing in.
            if is_dec_str:

                cur_node.children[0].children[0].children[0].children[0].content_string = repr('')



            # 🟩 visit node: find direct import
            linked_names = find_import_linked_names(cur_node)
            import_linked_names.update(linked_names)

            # 🟩 visit node: find secondary linked to import
            is_SCB, cls_name, the_cont_seq, the_prime_str = collect_self_complete_block_info(cur_node)
            if is_SCB:
                cls_names = [x[1] for x in scb_visit_list]
                conts = [x[2] for x in scb_visit_list]
                conts = list(map(lambda x: '' if x=='' else eval(x), conts))
                primes = [x[3] for x in scb_visit_list]
                for short_i in range(len(scb_visit_list)):
                    if conts[short_i] == '':
                        continue
                    if cls_names[short_i] != 'Name':  # discard all 'Name', because the 'Name' must itself be linked with Import.
                        def find_prev_adjacent_entity(i2):
                            i1 = i2-1
                            has_seen_keywords = False
                            while i1>=0:
                                if cls_names[i1] == 'keyword':
                                    has_seen_keywords = True
                                def is_imported_body(ibody):
                                    if ('Import' not in cls_names[0]) and (conts[ibody] not in import_linked_names):
                                        return False
                                    if cls_names[ibody] == 'Name':
                                        return True
                                    elif primes[ibody-1] == 'name=':
                                        return True
                                    else: 
                                        return False
                                if is_imported_body(i1):
                                    if not has_seen_keywords:
                                        break
                                    else:  # in this case,keep move backward, until see 'call', then move forward to find the first 'name'.
                                        while primes[i1] != 'Call(':
                                            i1 -= 1
                                        while cls_names[i1] != 'Name':
                                            i1 += 1
                                        break
                                else:
                                    i1 -= 1
                            return i1
                        nidx = find_prev_adjacent_entity(short_i)
                        if (nidx >=0) and (conts[nidx]!='') and (conts[nidx] in import_linked_names):
                            import_linked_names.update([conts[short_i]])
                scb_visit_list = []
                scb_visit_list.append([is_SCB, cls_name, the_cont_seq, the_prime_str])
            else:
                scb_visit_list.append([is_SCB, cls_name, the_cont_seq, the_prime_str])


            # 🟩 visit node finish, return to traversal.
            stack.append(cur_node)


            # stop grouping case 1/2: made non-trivial children selection only once. 
            if len(cur_node.children)>1:
                split_cnt += 1
        cur_content = cur_node.content_string

        syntax_sequence.append(cur_syntax)

        cur_syntax = ''
        content_sequence.append(cur_content)

    syntax_sequence.append(cur_syntax)

    assert len(syntax_sequence) == len(content_sequence)+1
    assert syntax_sequence[-1]

    compresssed_syn, compressed_cont = [], []
    s_holder = []
    for s,c in zip(syntax_sequence, content_sequence):
        s_holder.append(s)
        if c:
            compresssed_syn.append(''.join(s_holder))
            compressed_cont.append(c)
            s_holder = []

    compresssed_syn.append(''.join(s_holder)+syntax_sequence[-1])

    return import_linked_names, compresssed_syn, compressed_cont







def segment_then_put_in_template(devotion_list, chunk_lens, target_lens, template_list):
    # if devotion_list = '12345', chunk_lens(devotions) = [2,3], target_lens = [4,5], templace_list = 'xxxxxxxxx', then
    # output = 'xx12xx345'

    if (len(chunk_lens)!=len(target_lens)) or (len(devotion_list)!=sum(chunk_lens)) or (len(template_list)!=sum(target_lens)):
        if VERBOSE:
            print("samples are not all following the same structure, Do irregular padding.")
        chunk_lens = [len(devotion_list)]
        target_lens = [len(template_list)]

    filled = copy.deepcopy(template_list)
    def cum_sum(l1):
        l1 = list(l1)
        sumed = [0]
        while l1:
            sumed.append(sumed[-1] + l1.pop(0))
        return sumed
    acc = cum_sum(target_lens)
    acc0 = cum_sum(chunk_lens)
    for i in range(len(chunk_lens)):
        start, end = acc[i], acc[i+1]
        start0, end0 = acc0[i], acc0[i+1]
        filled[end-chunk_lens[i]:end] = devotion_list[start0:end0]
    return filled




def pre_api_int2token(int_seq, decoder, drop_cross_instance_pad_token=0, drop_cross_sample_pad_token=0):
    synSeq, contSeq = tonp(int_seq).T.tolist()

    syn_tokens = [decoder(i) for i in synSeq]
    cont_tokens = [decoder(i) for i in contSeq]
    # drop the cross instance padding
    if drop_cross_instance_pad_token:
        while syn_tokens and syn_tokens[-1]==vocabulary_defs.cross_instance_pad_syn_token:
            assert cont_tokens[-1]==vocabulary_defs.cross_instance_pad_cont_token, 'There are [Syntax Transformer Defined] syntax errors in the data'
            syn_tokens.pop()
            cont_tokens.pop()
    if syn_tokens==[]:
        assert cont_tokens==[], 'There are [Syntax Transformer Defined] syntax errors in the data'
        return [], []

    if drop_cross_sample_pad_token:
        # drop the cross sample padding
        _syn_tokens, _cont_tokens = syn_tokens, cont_tokens
        syn_tokens, cont_tokens = [], []
        for i in range(len(_syn_tokens)):
            if _syn_tokens[i]==vocabulary_defs.input_cross_sample_pad_syn_token:
                assert _cont_tokens[i]==vocabulary_defs.input_cross_sample_pad_cont_token, f'There are [Syntax Transformer Defined] syntax errors in the data, syn and cont padding does not match: {_syn_tokens[i]}, {_cont_tokens[i]}'
            else:
                syn_tokens.append(_syn_tokens[i])
                cont_tokens.append(_cont_tokens[i])

    poped = cont_tokens.pop()
    assert poped in [vocabulary_defs.content_final_chasing_syntax_token, vocabulary_defs.input_cross_sample_pad_cont_token, vocabulary_defs.cross_instance_pad_cont_token], 'There are [Syntax Transformer Defined] syntax errors in the data'
    return syn_tokens, cont_tokens

def pre_api_token_to_repr(syn_tokens, cont_tokens):
    assert len(syn_tokens) == len(cont_tokens)+1, 'There are errors in the data'
    synSeq2, contSeq2 = [], []
    for i in range(len(cont_tokens)):  # drop all padding tokens
        if (cont_tokens[i]!=vocabulary_defs.input_cross_sample_pad_cont_token) and \
            (syn_tokens[i]!=vocabulary_defs.input_cross_sample_pad_syn_token):
            if syn_tokens[i]!=vocabulary_defs.syntax_on_hold_for_content_token:
                synSeq2.append(syn_tokens[i])
                contSeq2.append(cont_tokens[i])
            else:
                synSeq2.append('')
                contSeq2.append(cont_tokens[i])
        else:
            assert (cont_tokens[i]==vocabulary_defs.input_cross_sample_pad_cont_token), 'There are errors in the data'
            assert (syn_tokens[i]==vocabulary_defs.input_cross_sample_pad_syn_token), 'There are errors in the data'
    synSeq2.append(syn_tokens[-1])

    contSeq2 = [repr(x) for x in contSeq2]

    synTree_str = step3_glue_context_content_seqs(synSeq2, contSeq2)
    recov_repr = step4_convert_entire_synTree_str_to_code_str(synTree_str)
    return recov_repr





