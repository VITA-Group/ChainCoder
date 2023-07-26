import transformers
import numpy as np

from tokenizer.tokenization_algorithm import (
    vocabulary_defs, 
    python_repr_tokenizer, 
    pre_api_token_to_repr,
    pre_api_int2token,
    tonp,
    load_txt,
    closest,
    segment_then_put_in_template,
    robust_tokenization_after_fix_vocab
)

VERBOSE = False


# 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 
# 🟩 For external useage, only use the following 12 functions, to avoid complicated repr()/eval() conversion.
# 🟩 I/O means: transformer input/output side, which are iodata/program
# 🟩 R,T,N means: 
# 🟩 R: the human readable obj (not exactly repr); for program it is string repr, for io data it is actually python Obj, NOT string repr.
# 🟩 T: tokens, len(syn_tokens) = len(cont_tokens) + 1
# 🟩 N: integer encoded tokens, equal length.
# 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 
def tokenizerAPI_IR2T(py_obj):
    py_repr = repr(py_obj)
    syn_tokens, cont_tokens = python_repr_tokenizer(py_repr, is_iodata=True)
    return syn_tokens, cont_tokens
def tokenizerAPI_OR2T(py_repr, coarse_level=False):
    if not robust_tokenization_after_fix_vocab:
        syn_tokens, cont_tokens = python_repr_tokenizer(py_repr, is_iodata=False, coarse_level=coarse_level)
        return syn_tokens, cont_tokens
    else:
        try:
            syn_tokens, cont_tokens = python_repr_tokenizer(py_repr, is_iodata=False, coarse_level=coarse_level)
            return syn_tokens, cont_tokens
        except:
            return [], []
def tokenizerAPI_IT2R(syn_tokens, cont_tokens):
    recov_repr = pre_api_token_to_repr(syn_tokens, cont_tokens)
    recov_obj = eval(recov_repr, {'inf': float('inf'), 'nan': float('nan')})
    return recov_obj
def tokenizerAPI_OT2R(syn_tokens, cont_tokens):
    recov_repr = pre_api_token_to_repr(syn_tokens, cont_tokens)
    return recov_repr
def tokenizerAPI_IT2N(syn_tokens, cont_tokens):
    cont_tokens += [vocabulary_defs.content_final_chasing_syntax_token]
    syn_intSeq = [vocabulary_defs.token2int_I(t, 'syn') for t in syn_tokens]
    cont_intSeq = [vocabulary_defs.token2int_I(t, 'cont') for t in cont_tokens]
    int_seq = np.array(list(zip(syn_intSeq, cont_intSeq))).tolist()
    return int_seq
def tokenizerAPI_OT2N(syn_tokens, cont_tokens, coarse_level=False):
    cont_tokens += [vocabulary_defs.content_final_chasing_syntax_token]
    if coarse_level:
        syn_intSeq = [vocabulary_defs.token2int_O(t, 'syn') for t in syn_tokens[1:]]
        cont_intSeq = [vocabulary_defs.token2int_O(t, 'cont') for t in cont_tokens[1:]]
        int_seq = np.array(list(zip(syn_intSeq, cont_intSeq))).tolist()
        
        syn_intSeq = [vocabulary_defs.token2int_O(t, 'syn') for t in syn_tokens[0]]
        cont_intSeq = [vocabulary_defs.token2int_O(t, 'cont') for t in cont_tokens[0]]
        int_seq_c = np.array(list(zip(syn_intSeq, cont_intSeq))).tolist()
        int_seq.insert(0, int_seq_c)
    else:
        syn_intSeq = [vocabulary_defs.token2int_O(t, 'syn') for t in syn_tokens]
        cont_intSeq = [vocabulary_defs.token2int_O(t, 'cont') for t in cont_tokens]
        int_seq = np.array(list(zip(syn_intSeq, cont_intSeq))).tolist()
    return int_seq
def tokenizerAPI_IN2T(int_seq, drop_cross_instance_pad_token=0, drop_cross_sample_pad_token=0):
    decoder = vocabulary_defs.int2token_I
    syn_tokens, cont_tokens = pre_api_int2token(int_seq, decoder, drop_cross_instance_pad_token, drop_cross_sample_pad_token)
    return syn_tokens, cont_tokens
def tokenizerAPI_ON2T(int_seq, drop_cross_instance_pad_token=0, drop_cross_sample_pad_token=0):
    decoder = vocabulary_defs.int2token_O
    syn_tokens, cont_tokens = pre_api_int2token(int_seq, decoder, drop_cross_instance_pad_token, drop_cross_sample_pad_token)
    return syn_tokens, cont_tokens
def tokenizerAPI_IN2R(int_seq):
    syn_tokens, cont_tokens = tokenizerAPI_IN2T(int_seq)
    recov_repr = tokenizerAPI_IT2R(syn_tokens, cont_tokens)
    return recov_repr
def tokenizerAPI_ON2R(int_seq):
    syn_tokens, cont_tokens = tokenizerAPI_ON2T(int_seq)
    recov_repr = tokenizerAPI_OT2R(syn_tokens, cont_tokens)
    return recov_repr
def tokenizerAPI_IR2N(py_repr):
    syn_tokens, cont_tokens = tokenizerAPI_IR2T(py_repr)
    int_seqs = tokenizerAPI_IT2N(syn_tokens, cont_tokens)
    return int_seqs
def tokenizerAPI_OR2N(py_repr):
    syn_tokens, cont_tokens = tokenizerAPI_OR2T(py_repr)
    int_seqs = tokenizerAPI_OT2N(syn_tokens, cont_tokens)
    return int_seqs
def tokenizerAPI_ON2R_flatten(int_seq, coarse_level=False):
    """ Used for transformer 1D sequence output.
    """
    try:
        if coarse_level:
            int_seq = tonp(int_seq).reshape(-1).tolist()
            which_subseq = 0
            s3, s4 = [], []
            for n in int_seq:
                if n==vocabulary_defs.s1234_sep_id:
                    which_subseq += 1
                if which_subseq==3:
                    s3.append(n)
                if which_subseq==3:
                    s4.append(n)
            s3, s4 = s3[1:], s4[1:]
            transpose = lambda matrix: [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
            int_seq = transpose([s3,s4])
            recov_repr = tokenizerAPI_ON2R(int_seq)
            return recov_repr
        else:
            int_seq = tonp(int_seq).reshape(-1)
            l = len(int_seq)
            int_seq = int_seq.reshape(l//2,2).tolist()
            recov_repr = tokenizerAPI_ON2R(int_seq)
            return recov_repr
    except:
        return None
def tokenizerAPI_D2N(txt_str):
    """ Convert from description texts to ints; paired with APPS pre-trained model.
    """
    tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
    int_list = tokenizer.encode(txt_str, verbose=False)
    return int_list









# 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 
# 🟩 other APIs.
def cross_sample_pad_aligning_syntax(iodata_ist2):
    """ padding the io data to align across different samples. The inputs/outputs of this function are all integer encoded sequence. For outputs, once the lengths are aligned, convert each sample into a numpy array - this is the best place to do so.
    The algorithm is:
        first convert back to obj, decouple back to i/o, then for each elem in i/o, again do tokenize, then, get the lengths of content tokens of each elem, pad with the longest one for every sample.
    Args:
        iodata_ist2: a nested list with int values coded tokens at the inner. Shape: [instance, sample, xy tokens, syn or cont subtokens]
    Returns:
        same shape list, but padded across sample (instance len are still different.)
    """
    
    cont_pad_id = vocabulary_defs.input_cross_sample_pad_cont_id
    syn_pad_id = vocabulary_defs.input_cross_sample_pad_syn_id

    nums_that_samples_give_to_each_placeholder = [] # this is a irregular shaped 3D list with shape: [N_instance, N_sample, N_placeholder in this instance]. For each sample, this variable stores how many content token it gives to each placeholders as specified by this instance.
    sample_seqlens = []
    for instance in iodata_ist2:
        nums_that_samples_give_to_each_placeholder.append([])
        tmp = []
        for sample_intSeq in instance:
            tmp.append(len(sample_intSeq))
            nums_that_samples_give_to_each_placeholder[-1].append([])
            inp, outp = tokenizerAPI_IN2R(sample_intSeq) # this is a list with shape N_sample x 2 (2 is the io)

            for body in list(inp) + list(outp):
                retok_syn, retok_cont = tokenizerAPI_IR2T(body)
                nums_that_samples_give_to_each_placeholder[-1][-1].append(len(retok_cont))

        sample_seqlens.append(max(tmp))

    def irregular_pad(data_st2, pad_target_len):
        res = []
        for t2 in data_st2:
            res.append( t2 + [[syn_pad_id, cont_pad_id]] * (pad_target_len-len(t2)) )
        return res

    res_ISt2 = []
    for inst_id, instance in enumerate(iodata_ist2):
        res_ISt2.append([])
        try: # normally it would be padded according to syntax roles here
            sample_devotions = np.asarray(nums_that_samples_give_to_each_placeholder[inst_id])
            max_max = sample_seqlens[inst_id]-1
            max_devotions = sample_devotions.max(axis=0) # this is 1-D array with shape [num_placeholders] specified by this instance. It contain int values, meaning the padding target lens for each syntax role in this instance.
            if sum(max_devotions)<max_max:
                if VERBOSE:
                    print("samples are not all following the same structure, Do irregular padding.")
                max_devotions = [max_max]
            else:
                if VERBOSE:
                    print('samples are not... emm, good padding')


            for i_samp, (sample, devotions) in enumerate(zip(instance, sample_devotions)):
                yet_to_fill_orig_values = [[syn_pad_id, cont_pad_id] for _ in range(sum(max_devotions))]
                yet_to_fill_orig_values[-1] = sample[-1]
                if sum(devotions) == len(sample)-1 and VERBOSE:
                    print("My implementation shouldn't be wrong, check your inputs, are all samples following the same structure?")  # num of content tokens on both side
                filled = segment_then_put_in_template(sample[:-1], devotions, max_devotions, yet_to_fill_orig_values) + [sample[-1]]
                res_ISt2[-1].append(np.array(filled).astype(int))
            res_ISt2[-1] = np.array(res_ISt2[-1]).astype(int)
        except:
            res_ISt2[-1] = irregular_pad(instance, sample_seqlens[inst_id])

        # assert len(res_ISt2[-1])>0


    for i, instance in enumerate(res_ISt2):
        res_ISt2[i] = np.array(instance).astype(int)
        # assert len(res_ISt2[i])>0

    return res_ISt2

def cross_instance_pad_io(inp_ist2):
    """ Pad the io data across instance, to make sure 1. the num sample are the same, 2. the sample length are the same across samples. The inputs/outputs of this function are all integer encoded sequences.
    Args:
        inp_ist2: a irregular list with np.array (i=list, st2=np.array), shape = [instance, sample, token, syn & cont]
    Returns:
        padded_ist2: np.array, shape = [instance, sample, token, syn & cont]
    """
    def fit_small_into_large_3D(small_arr, large_arr):
        d0,d1,d2 = small_arr.shape
        large_arr[:d0, :d1, :d2] = small_arr
        return large_arr
    Ni = len(inp_ist2)
    cross_inst_pad_syn_id = vocabulary_defs.input_cross_instance_pad_syn_id
    cross_inst_pad_cont_id = vocabulary_defs.input_cross_instance_pad_cont_id

    # 🟩 pad for both sample and token len
    Ns = max([x.shape[0] for x in inp_ist2])
    Nt = max([x.shape[1] for x in inp_ist2])

    padded_ist2 = np.zeros([Ni,Ns,Nt,2])
    padded_ist2[..., 0] = cross_inst_pad_syn_id
    padded_ist2[..., 1] = cross_inst_pad_cont_id
    token_lens_per_inst = []
    samp_lens_per_inst = []
    for inst_id in range(Ni):
        st2 = inp_ist2[inst_id]
        num_samp, max_tok_len, _ = st2.shape
        token_lens_per_inst.append(max_tok_len)
        samp_lens_per_inst.append(num_samp)
        padded_ist2[inst_id] = fit_small_into_large_3D(st2, padded_ist2[inst_id])

    return padded_ist2, samp_lens_per_inst, token_lens_per_inst



def cross_instance_pad_code_interleaved(inp_it2):
    """ Pad the code data across instance.
    Difference with `cross_instance_pad_io`: this one will add BOS/EOS
    """

    Ni = len(inp_it2)
    pad_id = vocabulary_defs.output_pad_id
    bos = np.array([vocabulary_defs.bos_token_id])
    eos = np.array([vocabulary_defs.eos_token_id])

    # 🟩 pad for both sample and token len
    Nt = max([len(x) for x in inp_it2])

    padded_it = np.zeros([Ni,Nt*2 + 2]) + pad_id  # +2 is for BOS and EOS
    lens = []

    for inst_id in range(Ni):
        t2 = np.concatenate([bos, tonp(inp_it2[inst_id]).reshape(-1), eos], axis=0)
        padded_it[inst_id][:len(t2)] = t2
        lens.append(len(t2))

    return padded_it, lens


def cross_instance_pad_code(inp_it2):
    """ Pad the code data across instance.
    As defined in get_ChainCoder_dataloader in sttd.py, inp_it2[instance_id] has the structure of:
        [ [[coarse_s0, coarse_c0], [coarse_s1, coarse_c1], ... ] ,  [s0,c0], [s1,c1], [s2,c2], ...]
    """


    coarse_seq = [t2[0] for t2 in inp_it2]
    inp_it2 = [t2[1:] for t2 in inp_it2]
    # inp_it2 = inp_it2[1:]

    Ni = len(inp_it2)
    pad_id = vocabulary_defs.output_pad_id
    ssep = [vocabulary_defs.s1234_sep_id]
    
    bos = np.array([vocabulary_defs.bos_token_id])
    eos = np.array([vocabulary_defs.eos_token_id])

    # 🟩 pad for both sample and token len
    Nt = max([len(x)+len(y) for x,y in zip(coarse_seq, inp_it2)])

    padded_it = np.zeros([Ni, Nt*2 + 2 + 3]) + pad_id  # +2 is for BOS and EOS; +3 for s1234 sep
    lens = []

    for inst_id in range(Ni):
        s1, s2 = tonp(coarse_seq[inst_id]).transpose(1,0).tolist()
        s3, s4 = tonp(inp_it2[inst_id]).transpose(1,0).tolist()
        
        t2 = np.concatenate([bos,s1,ssep,s2,ssep,s3,ssep,s4,eos], axis=0)
        padded_it[inst_id][:len(t2)] = t2
        lens.append(len(t2))

    return padded_it, lens

