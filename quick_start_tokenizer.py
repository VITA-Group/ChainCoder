import os
from tokenizer.tokenizerAPI import (
    tokenizerAPI_IR2T,
    tokenizerAPI_OR2T,
    tokenizerAPI_IT2R,
    tokenizerAPI_OT2R,
    tokenizerAPI_IT2N,
    tokenizerAPI_OT2N,
    tokenizerAPI_IN2T,
    tokenizerAPI_ON2T,
    tokenizerAPI_IN2R,
    tokenizerAPI_ON2R,
    tokenizerAPI_IR2N,
    tokenizerAPI_OR2N,
)

    
def myprint(*x):
    '''
    print both to terminal and an offline file
    '''
    print(*x)
    print(*x, file=open('quick_start_tokenizer_output.txt', 'a'))

sep = '\n\n________________________________\n'



example_code_string_representation = '''
# Leetcode problem 7. Reverse Integer: 
# https://leetcode.com/problems/reverse-integer/description/
class Solution(object):
    def reverse(self, x):
        reverse = 0
        sign = -1 if x < 0 else 1
        x = abs(x)
        while x:
            digit = x % 10
            reverse = reverse * 10 + digit
            x /= 10
        result = sign * reverse
        if result > 2 ** 31 - 1 or result < -(2 ** 31):
            return 0
        return result
'''
# example_io_data = [[1,4,'hello'], [True,False]]
example_io_data = [1,4,'hello']

if os.path.exists('quick_start_tokenizer_output.txt'):
    os.remove('quick_start_tokenizer_output.txt')


myprint(f'{sep}demo code:\n{example_code_string_representation}')

# ---- Visualization for: python code string -> S3 and S4 subsequences
syntax_token_S3, content_tokens_S4 = tokenizerAPI_OR2T(example_code_string_representation)
myprint(f'{sep}syntax_token_S3:')
for x in syntax_token_S3:
    myprint(x)
myprint(f'{sep}content_tokens_S4:')
for x in content_tokens_S4:
    myprint(x)


# ---- Visualization for: S3/S4 -> integer sequence
int_seq = tokenizerAPI_OT2N(syntax_token_S3, content_tokens_S4)
myprint(f'{sep}integer sequence:\n{int_seq}')

# ---- Visualization for combined one-step: python code string -> integer sequence
int_seq = tokenizerAPI_OR2N(example_code_string_representation)
myprint(f'{sep}integer sequence:\n{int_seq}')

# ---- convert back to python string
py_code_string = tokenizerAPI_ON2R(int_seq)
myprint(f'{sep}Python code converted back:{py_code_string}')

# ---- Visualization for: io_obj -> token represnetations
myprint(f'{sep}I/O python object:\n{example_io_data}')
syn_IO, cont_IO = tokenizerAPI_IR2T(example_io_data)
myprint(f'{sep}IO data tokens:\n{syn_IO}\n{cont_IO}\n')

# ---- Visualization for: IO data -> int sequence -> convert back to IO data
# io_data_back = tokenizerAPI_IN2R(tokenizerAPI_IR2N(example_io_data))
# myprint(f'{sep}IO data converted back:\n{io_data_back}')


os.system(f"open quick_start_tokenizer_output.txt")

