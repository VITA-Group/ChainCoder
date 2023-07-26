import itertools
from typing import List, Tuple
from collections import Counter, defaultdict
import copy
import re
import numpy as np
from joblib import Parallel, delayed



from dataloaders.check_exec_match import convert_obj_back_to_raw, forward_run_code, check_io_match_one_sample_obj
from dataloaders.loader_utils import evalio, shuffled


MAX_LST1D_LEN = 256

def lst_find_elem(arr, b):
    arr = np.array(arr).reshape(-1)
    where = np.where(arr==b)[0][0]
    return where


def iodata_augmentor(code_raw_st, io_s2t_orig, required_size, one_sample_exec_timeout, args):

    def process(x, code):
        raw_x = convert_obj_back_to_raw(x)
        
        y = forward_run_code(raw_x, code, one_sample_exec_timeout)
        
        return y if not (type(y) is RuntimeError) else None

    def decide_if_need_dist_control():
        do_dist_control = False
        N_prob = 1
        distinct_elems = None
        min_cnt = required_size

        if io_s2t_orig==[] or io_s2t_orig[0]==[]:
            do_dist_control = False
        else:
            ls2d = io_s2t_orig[0][1] # check odata of the first sample
            ls2d = itertools.chain(*ls2d)
            for i, ix in enumerate(ls2d):
                if type(ix) is bool:
                    do_dist_control = True
                    distinct_elems = [True, False]

                elif type(ix) is str:
                    if ('yes' in ix.lower()) or ('no' in ix.lower()):
                        distinct_elems = ['yes', 'no']
                        do_dist_control = True
                        break

                    elif ('true' in ix.lower()) or ('false' in ix.lower()):
                        distinct_elems = ['true', 'false']
                        do_dist_control = True
                        break

            if do_dist_control:
                N_prob = 2
                min_cnt = required_size//len(distinct_elems)
                if min_cnt==0:
                    print(f'in iodata_augmentor, Requested samples per instance ({required_size}) < categories ({len(distinct_elems)}). This is a bad choice, consider making samples_per_instance bigger. ')


        print(f'in iodata_augmentor, do_dist_control = {do_dist_control}')
        return do_dist_control, N_prob, distinct_elems, min_cnt

    def min_len_of_dic(dic):
        lens = [len(dic[x]) for x in distinct_elems]
        return min(lens)
    def grab(idata, odata, min_cnt):
        res = []
        for k in distinct_elems:
            poped = idata[k][:min_cnt]
            idata[k] = idata[k][min_cnt:]

            poped2 = odata[k][:min_cnt]
            odata[k] = odata[k][min_cnt:]

            res.extend(list(zip(poped, poped2)))
        resti = list(itertools.chain.from_iterable(idata.values()))
        resto = list(itertools.chain.from_iterable(odata.values()))
        rest = list(zip(resti, resto))
        np.random.shuffle(rest)
        if required_size-len(res)<=len(rest):
            res.extend(rest[:required_size-len(res)])
        else:
            res.extend(rest)
            ridx = np.random.choice(len(res), len(rest) - (required_size-len(res)))
            res.extend([res[r] for r in ridx])
        assert len(res)==required_size
        np.random.shuffle(res)
        aligned_i, aligned_o = list(zip(*res))
        return aligned_i, aligned_o



    def try_generate(io_s2t_orig, N_prob, idata, odata, batch_size_when_one_probe, code_pass_cnts=None):
        """
        code_pass_cnts: [code_fail_orig_label, code_pass_orig_label, code_fail_new_data, code_pass_new_data]
        """

        len_o = 0

        orig_idata_s2t = [x[0] for x in io_s2t_orig]
        orig_odata_s2t = [x[1] for x in io_s2t_orig]
        do_orig_code_check = (code_pass_cnts is not None)

        if do_orig_code_check:
            code_pass_cnts[:2] = check_match(orig_idata_s2t, code_pass_cnts[:2])
            print(f'After checking this instance codematch, code_pass_cnts = {code_pass_cnts}')

        for i_prob in range(N_prob):

            augmented_idata_s2t = end_to_end_idata_sat_augment(orig_idata_s2t, batch_size_when_one_probe)


            # print('Warning: In symbolic_exec.py, if set n_jobs to > 2 * samples_per_instance, you may end up nothing! If too small, will not lose your instances, but much slower. If set to 1, you may not see any print. Recommended 0.5*samples_per_instance.')

            if args.verbose:
                print(f'in data_augmentation.py, starting forward run: \n\t size = {batch_size_when_one_probe}\n\t valid outputs so far = {len_o}\n\t required_size = {required_size} \n\t prob progress: {i_prob} / {N_prob}\n\t one_sample_exec_timeout = {one_sample_exec_timeout}')

            code = np.random.choice(code_raw_st)


            # ---- option 1: parallel
            ys = Parallel(n_jobs=64)(delayed(process)(x) for x in augmented_idata_s2t)
            # ys = Parallel(n_jobs=2)(delayed(process)(x) for x in augmented_idata_s2t)
            # ys = Parallel(n_jobs=1)(delayed(process)(x) for x in augmented_idata_s2t)
            # ys = Parallel(n_jobs=len(augmented_idata_s2t)//2)(delayed(process)(x) for x in augmented_idata_s2t)

            # ys = Parallel(n_jobs=batch_size_when_one_probe//2)(delayed(process)(x) for x in augmented_idata_s2t)

            # ---- option 2: no parallel
            # ys = [] 
            # for x in augmented_idata_s2t:
            #     y = process(x, code)
            #     ys.append(y)


            if args.verbose:
                print(f'in data_augmentation.py,  {i_prob} / {N_prob} prob finish. ')


            indicies = [i for i, y in enumerate(ys) if y is not None]
            i_data = [augmented_idata_s2t[i] for i in indicies]
            o_data = [ys[i] for i in indicies]
            len_o += len(o_data)

            if not do_dist_control:
                idata.extend(i_data)
                odata.extend(o_data)
            else:
                for isample, osample in zip(i_data, o_data):
                    for tin in shuffled(distinct_elems):
                        if tin in osample.lower():
                            idata[tin].append(isample)
                            odata[tin].append(osample)
                            break
                if min_len_of_dic(odata)>=min_cnt:
                    break
        
        print(f'in try_generate, requested to augment {N_prob} probs x {batch_size_when_one_probe} batch of = {N_prob*batch_size_when_one_probe} idata; when return, actually looped {i_prob+1} x {batch_size_when_one_probe} = {(i_prob+1)*batch_size_when_one_probe} samples, and got {len_o} odata passed.')
        print(odata)

        return idata, odata



    # 🟩 begin run
    do_dist_control, N_prob, distinct_elems, min_cnt = decide_if_need_dist_control()

    if do_dist_control:
        idata = {x: [] for x in distinct_elems}
        odata = {x: [] for x in distinct_elems}

    else:
        idata, odata = [], []
    ueven = lambda n: (int(n)//2)*2+2


    batch_size_when_one_probe = ueven(required_size)

    code_pass_cnts = [0,0,0,0]
    idata, odata = try_generate(io_s2t_orig, N_prob, idata, odata, batch_size_when_one_probe, code_pass_cnts)
    print

    # 🟩 the distribution control part
    if do_dist_control:
        _min = min_len_of_dic(odata)
        if _min<min_cnt:
            if _min==0: _min=0.2
            # new_tolerance = max(int(N_prob*min_cnt/_min*2), required_size*500)   # For this dist controlled instance, I can give you maximum (500x) longer time, to generate samples.
            new_tolerance = int(required_size*20)


            if args.verbose: print(f'in data_augmentation.py, After initial probs:\n\t Need New probs: YES.\n\t currently got samples: {_min} \n\t required_size = {required_size}\n\t do_dist_control: {do_dist_control} \n\t new_tolerance = {new_tolerance}\n\t ')

            idata, odata = try_generate(io_s2t_orig, new_tolerance, idata, odata, batch_size_when_one_probe)
            print

            if min_len_of_dic(odata)<min_cnt:
                print(f'Generated one "bad sample" in data_aug: {min_len_of_dic(odata), min_cnt}')
            # else:
            iAug_sat, oAug_sat = grab(idata, odata, min_cnt)

        else:
            iAug_sat, oAug_sat = grab(idata, odata, min_cnt)

    else:
        if len(idata)<required_size:
            _cur = 0.2 if len(idata)==0 else len(idata)
            # new_tolerance = max( int(N_prob*required_size/_cur*2), required_size*10)
            new_tolerance = int(required_size*10)

            if args.verbose: print(f'in data_augmentation.py, After initial probs:\n\t Need New probs: YES.\n\t currently got samples: {_cur} \n\t required_size = {required_size}\n\t do_dist_control: {do_dist_control} \n\t new_tolerance = {new_tolerance}\n\t ')

            idata, odata = try_generate(io_s2t_orig, new_tolerance, idata, odata, batch_size_when_one_probe)
            print

            if len(idata)<=required_size:
                iAug_sat, oAug_sat = [], []

            else:
                iAug_sat, oAug_sat = idata[:required_size], odata[:required_size]
        else:
            iAug_sat, oAug_sat = idata[:required_size], odata[:required_size]

    print(f'at the end of iodata_augmentor:\n\t iAug_sat = {iAug_sat} \n\t oAug_sat = {oAug_sat}\n\t originally shown {len(io_s2t_orig[0])} samples\n\t requested {required_size} augmentation target\n\t actually generated {len(oAug_sat)} odata\n\t do_dist_control = {do_dist_control}\n\t ')

    return iAug_sat, oAug_sat





def all_equal(lst):
    if len(lst)==0:
        
        print('\n all_equal function met rare case of empty input, not expected... \n')
        raise
        return True
    a = lst[0]
    for i in range(1, len(lst)):
        if a != lst[i]:
            return False
    return True


def end_to_end_idata_sat_augment(idata_sat, required_size):
    # 🟩 step1: for each sample, get the okType, AND templates
    def get_per_sample_lst1D_type_list(lst2D):
        """ Assume lst1D is homo inside, then For each lst1D elem, return : [dtype in this lst1D, is it len==1]
        """
        type_lst = []
        for lst1D in lst2D:
            try:
                assert all_equal([type(x) for x in lst1D]), [type(x) for x in lst1D]
            except:
                print([type(x) for x in lst1D])
            if len(lst1D)>1:
                possibly_master_value = None
            elif len(lst1D)==1:
                possibly_master_value = lst1D[0]
            thistype = (lst1D[0].__class__.__name__, len(lst1D)==1, possibly_master_value, len(lst1D), lst1D[0])
            type_lst.append(thistype)

        return type_lst


    def dismantle(input_list):
        
        group_len = 1
        k = 1
        def two_heter_equal(sub_lst1, sub_lst2):
            sub_lst1 = [lst1DType[:2] for lst1DType in sub_lst1] # lst1DType is: [type class name, len_is_1, possibly_master_value]
            sub_lst2 = [lst1DType[:2] for lst1DType in sub_lst2]
            return sub_lst1==sub_lst2

        for group_len in range(1, len(input_list) + 1):
            if k != 1:
                break
            last_group = input_list[-group_len:]
            k = 1
            for i in range(group_len, len(input_list), group_len):
                intermediate_group = input_list[-i - group_len : -i]
                if not two_heter_equal(intermediate_group, last_group):
                    break
                else:
                    k += 1

        # if no groups found
        if k == 1 and group_len == len(input_list):
            k = 0
            group_len = 0
            pre_len = len(input_list)
            master_idx_guess = []
            heterogeneous_part = input_list

        # some group found
        else:
            # default reduce group len due to last loop increment
            group_len -= 1

            # crop heterogeneous part and find length
            heterogeneous_part = input_list[: -k * group_len]
            pre_len = len(heterogeneous_part)

            # guess master positions
            master_idx_guess = [(i, 'master_is_controling_the_k') for i, componets in enumerate(heterogeneous_part) if componets[2] == k]

        all_lens_of_lst1D = list(map(lambda x: x[3], heterogeneous_part))
        for i_pre in range(pre_len):
            len_of_lst1D = heterogeneous_part[i_pre][3]
            value_of_first = heterogeneous_part[i_pre][4]

            if len_of_lst1D==1 and (type(value_of_first) is int) and (value_of_first>1) and (value_of_first in all_lens_of_lst1D):  # if some heter list is [4], i.e., single element value>1, and this '4' is equal to the len of another lst1D in the heter part.
                who_is_being_controled = np.where(np.asarray(all_lens_of_lst1D)==value_of_first)[0][0]
                master_idx_guess.append((i_pre, f'master_is_controling_the_len_of_idx_{who_is_being_controled}_in_heter_part'))

        

        return pre_len, group_len, k, master_idx_guess

    # 🟩 Logic: you are given a list of samples. Each sample's type is uniqally represented by okType, which contains 4 parts:
        # opt_template: a list of strings
        # kgroup_template: a list of strings
        # k
        # master_idx_guess

    okTps_samp_hh_supp_3body = []
    raw_ks = []
    master_id_guess_list = []
    for idata_at in idata_sat:
        thatTriple_list = get_per_sample_lst1D_type_list(idata_at)
        pre_len, group_len, k, master_idx_guess = dismantle(thatTriple_list)
        master_id_guess_list.append(master_idx_guess)
        opt_template = list(thatTriple_list[:pre_len])
        kgroup_template = list(thatTriple_list[pre_len:pre_len+group_len])
        okTps_samp_hh_supp_3body.append((opt_template, kgroup_template))
        raw_ks.append(k)





    # 🟩 step2: vote and filter
    def vote_and_filter(idata_sat, okTps_samp_hh_supp_3body, master_id_guess_list):
        def most_frequent(input_list):
            input_list = [repr(x) for x in input_list]
            occurence_count = Counter(input_list)
            most = occurence_count.most_common(1)[0][0]
            return evalio(most)

        first_2_of_tripple = okTps_samp_hh_supp_3body
        for i_samp in range(len(okTps_samp_hh_supp_3body)):
            for i_hh in range(2):
                for i_supp in range(len(okTps_samp_hh_supp_3body[i_samp][i_hh])):
                    first_2_of_tripple[i_samp][i_hh][i_supp] = first_2_of_tripple[i_samp][i_hh][i_supp][:2]
        agreed_okType = most_frequent(first_2_of_tripple)



        idx = [i for i, x in enumerate(okTps_samp_hh_supp_3body) if x == agreed_okType]


        master_id_guess_list = [master_id_guess_list[i] for i in idx]
        master_id_meaning = None
        for guess_list in master_id_guess_list:
            if master_id_meaning is None:
                master_id_meaning = set(guess_list)
            else:
                master_id_meaning = master_id_meaning.intersection(set(guess_list))
        if len(master_id_meaning) != 0:
            master_id_meaning = list(master_id_meaning)[0]
        else:
            master_id_meaning = None




        # two options below, discard or not.
        agreed_idata_sat = [idata_sat[i] for i in idx]
        # agreed_samples = idata_sat


        agreed_pre_template, agreed_kgroup_template = agreed_okType

        agreed_pre_len, agreed_group_len = len(agreed_pre_template), len(agreed_kgroup_template)

        return agreed_idata_sat, master_id_meaning, (agreed_pre_len, agreed_group_len, agreed_pre_template, agreed_kgroup_template)

    agreed_idata_sat, agreed_master_id_meaning, (agreed_pre_len, agreed_group_len, agreed_pre_template, agreed_kgroup_template) = vote_and_filter(idata_sat, okTps_samp_hh_supp_3body, master_id_guess_list)
    
    orig_idata_sat = idata_sat

    del pre_len, group_len, opt_template, kgroup_template, idata_sat, okTps_samp_hh_supp_3body, master_id_guess_list

    


    # 🟩 step3: grab all agreed data

    def grab_demo_from_samples():

        agreed_datas_heter = list(map(list, list(iter(zip(*agreed_idata_sat)))))[:agreed_pre_len]

        ks_orig = []
        for idata_at in agreed_idata_sat:
            if agreed_group_len==0:
                ks_orig.append(0)
            else:
                ks_orig.append((len(idata_at) - agreed_pre_len) // agreed_group_len)

        agreed_datas_homo = []
        for row in list(map(list, list(iter(zip(*agreed_idata_sat)))))[agreed_pre_len : agreed_pre_len + agreed_group_len]:
            temp = [[element] * k for k, element in zip(ks_orig, row)]
            agreed_datas_homo.append(list(itertools.chain(*temp)))
            

        return agreed_datas_heter, agreed_datas_homo, ks_orig

    agreed_datas_heter, agreed_datas_homo, ks_orig = grab_demo_from_samples()


    
    # 🟩 step4: infer secondary types: generate the 'final_tmpl_sat'

    def get_ks_infer():
        if all_equal(ks_orig):
            fixed_k = ks_orig[0]
            inferred_ks = [fixed_k for _ in range(required_size)]

            stil_mutate_len_prob = 0.05
            if np.random.rand()<stil_mutate_len_prob:  # with small prob, even when orig tells the 'k' value is fixed across samples, still mutate 'k'.
                variance = max(fixed_k//3, 1)
                inferred_ks = (np.array(inferred_ks) + np.random.randint(-variance, variance+1, required_size)).tolist()


        else:  # ks can vary length; so can also = 0.
            med = np.median(ks_orig)
            std = min(np.std(ks_orig), med)

            if np.random.rand()<0.2:  # with small prob, sample from gaussian ; mainly from uniform
                inferred_ks = np.random.normal(float(med), float(std), required_size)
            else:  # sample from uniform.
                upperbound = np.random.choice(int(med+3*std))
                upperbound = max(upperbound, 5) # if upper bound is too small, e.g., =1, then all ks will =0. Not desired.
                inferred_ks = np.random.choice(upperbound, size=required_size)


            inferred_ks = [int(x) for x in inferred_ks]


        for i in range(required_size): # make sure non-negative
            if inferred_ks[i]<0:  # reset to 0 or 1; but mostly reset to 0.
                inferred_ks[i] = int(np.random.rand()<0.1)

        return inferred_ks


    inferred_ks = get_ks_infer()

    def calculate_generator_support():
        """ The generator support is two lists of strings, the prev and the kgroup. It is shared across samples.
        """
        def cal_support(datas2D, tmpl_tuple):
            # assert len(datas2D)==len(type_list_first_round)
            # nums:     !fix_len:5! / !varing_len:6.5:4.21! / !master!
            # sign:  $p0n$ / $b01$ / $p$ / $p0$
            # absolute upperbound: ^10^ / ^100^ / ^1000^
            # distribution type:  ~uniformly~ / ~same~ / ~around~ / ~gaussian~
            def get_num(_datas_2D):
                all_lens = [len(x) for x in _datas_2D]
                median = np.median(all_lens)
                std = np.std(all_lens)
                if std==0.:
                    return f'!fix_len:{median}!'
                else:
                    return f'!varing_len:{median}:{std}!'
            def get_sign(_datas_2D):
                flattened = list(itertools.chain.from_iterable(_datas_2D))
                if len(set(flattened))==2 and sorted(set(flattened))==[0,1]:
                    return '$b01$'
                is_positive = [x>0 for x in flattened]
                is_p0 = [x>=0 for x in flattened]
                if all_equal(is_positive) and is_positive[0]==True:
                    return '$p$'
                elif all_equal(is_p0) and is_p0[0]==True:
                    return '$p0$'
                else:
                    return '$p0n$'
            def get_upperbound(_datas_2D):
                flattened = list(itertools.chain.from_iterable(_datas_2D))
                maxv = max([abs(x) for x in flattened])
                if maxv<10:
                    return '^10^'
                elif maxv<100:
                    return '^100^'
                else:
                    return '^1000^'

            raw_type, is_len_1 = tmpl_tuple
            if raw_type=='int':
                support = 'int|' + get_num(datas2D) + get_sign(datas2D) + get_upperbound(datas2D)
            elif raw_type=='str':
                support = 'str|' + get_num(datas2D)
            elif raw_type=='float':
                support = 'float|' + get_num(datas2D) + get_sign(datas2D) + get_upperbound(datas2D)
            else:
                raise NotImplementedError(raw_type)
            return support

        assert len(agreed_datas_heter)==len(agreed_pre_template)
        assert len(agreed_datas_homo)==len(agreed_kgroup_template)
        support_pre = []
        support_okgroup = []
        for collected_2D, templ_tuple in zip(agreed_datas_heter, agreed_pre_template):
            support_pre.append(cal_support(collected_2D, templ_tuple))
        for collected_2D, templ_tuple in zip(agreed_datas_homo, agreed_kgroup_template):
            support_okgroup.append(cal_support(collected_2D, templ_tuple))
        if agreed_master_id_meaning is not None:
            support_pre[agreed_master_id_meaning[0]] = 'int|' + '!master:1!' + '$p0$' + '^1000^' + f'>{agreed_master_id_meaning[1]}>'
        return support_pre, support_okgroup

    support_pre, support_okgroup = calculate_generator_support()


    # To provide secondary type API for next steps, first, ensure a shape-fixed template
    final_tmpl_sat = [[] for _ in range(required_size)]
    for i_samp in range(required_size):
        final_tmpl_sat[i_samp] = ['' for _ in range(agreed_pre_len)]
        for i_group in range(inferred_ks[i_samp]):
            final_tmpl_sat[i_samp] += ['' for _ in range(agreed_group_len)]

    # sedond, fill values
    for i_samp in range(required_size):
        for i_pre in range(agreed_pre_len):
            final_tmpl_sat[i_samp][i_pre] = support_pre[i_pre] + f'@SAMPLE{i_samp}@HOMOCNT{i_pre}'
        for i_k in range(inferred_ks[i_samp]):
            for i_group in range(agreed_group_len):
                this_arg_idx = agreed_pre_len + agreed_group_len * i_k + i_group
                final_tmpl_sat[i_samp][this_arg_idx] = support_okgroup[i_group] + f'@SAMPLE{i_samp}@HOMOCNT{agreed_pre_len + i_group}'  # each pre within one sample uses one unique generator; all kgroup share the same generator.

    # 🟩 step5: use secondary types to augment and generate final
    def replace_with_prob(orig, target, prob):
        if np.random.rand()<prob:
            return target
        else:
            return orig

    def generate_binary(list_length):
        this_len = decide_generated_len_of_lst1D(list_length)
        gen1D = np.random.choice([0, 1], this_len)
        return gen1D


    def decide_generated_len_of_lst1D(list_length):
        if list_length[0] == "fix_len": 
            
            this_len = int(eval(list_length[1]))
            assert this_len>0

            if np.random.rand()<0.02: # with very small prob, still vary it.
                this_len = int(np.random.normal(this_len, this_len*0.5))

        elif list_length[0] == "varing_len":

            def specifically_grab_thislen_from_uniform_with_bound(ub):
                this_len = np.random.choice(ub)
                return this_len

            _rand = np.random.rand()
            if _rand<0.2:
                this_len = specifically_grab_thislen_from_uniform_with_bound(10)
            elif 0.2<_rand<0.4:
                this_len = specifically_grab_thislen_from_uniform_with_bound(20)
            elif 0.4<=_rand<0.6:
                this_len = specifically_grab_thislen_from_uniform_with_bound(100)
            elif 0.6<=_rand<0.8:
                med, std = list_length[1].split(":")
                med, std = float(med), float(std)
                this_len = int(np.random.normal(med, std))
            elif 0.8<=_rand<1.0:
                med, std = list_length[1].split(":")
                med, std = float(med), float(std)
                med, std = med*10 , std*10
                this_len = int(np.random.normal(med, std))


        elif list_length[0] == "master":
            this_len = 1

            if np.random.rand()<0.005: # with very small prob, still vary it.
                this_len = np.random.choice([2,3,4])
        else:
            raise NotImplementedError

        this_len = max(this_len, 1) # ensure lst1D has at least 1 elem

        if (this_len > MAX_LST1D_LEN):  # too long inputs take much more GPU memory. Limit it.
            this_len = np.random.choice(20) + MAX_LST1D_LEN

        return this_len


    def gen_ints(gen_name, list_length, sign, prev_bound, random_func=np.random.randint, this_k=None):
        is_master = False
        pending_master = False

        gen_len = decide_generated_len_of_lst1D(list_length)
        if list_length[0] == "master":
            assert this_k != None
            is_master = True

        # decide the value distribution
        if is_master:
            low = this_k
            high = this_k + 1
        else:

            assert prev_bound in [10, 100, 1000]
            ub_of_bound = [10, 30, 100, 300, 1000, 3000] # with small prob, change original bound
            probs = [1/8]*6
            probs[lst_find_elem(ub_of_bound, prev_bound)] *= 3

            bound = max(3, np.random.choice(ub_of_bound, p=probs))  # bound shall not be too small.

            if sign == "p0n":
                low = -bound + 1
                high = bound
            elif sign == "p0":
                low = 0
                high = bound
            elif sign == "p":
                low = 1
                high = bound
            else:
                raise NotImplementedError

        distri_types = [
            'uniform', 'gaussian', 
            'uniform_in_some_interval',
            'all_equal', 'almost_all_equal',
            'range_from_0', 'range_from_sth_middle',
            ]
        distri_probs = [
            0.7, 0.0,
            0.0, 
            0.1, 0.2,
            0., 0.0,
            ]
        dist = np.random.choice(distri_types, 1, p=distri_probs)
        if gen_len < 0: gen_len=0
        if dist == "uniform":
            gen1D = random_func(low, high, gen_len)
        elif dist in ["all_equal", "almost_all_equal"]:
            num = random_func(low, high)
            gen1D = np.repeat(num, gen_len)
            if dist == "almost_all_equal":
                gen1D += random_func(-2, 3, gen_len)
        else:
            raise NotImplementedError


        gen1D = gen1D.tolist()


        if is_master:
            if np.random.rand()<0.995: # with very high prob, stick to master.
                master_mean = re.findall(r"\>(.*?)\>", gen_name)[0]
                if master_mean=='master_is_controling_the_k':
                    gen1D = [this_k]
                elif master_mean.startswith('master_is_controling_the_len_of_idx_'):
                    master_mean = master_mean.split('master_is_controling_the_len_of_idx_')[-1].split('_in_heter_part')[0]
                    pending_id = eval(master_mean)
                    gen1D = [ValueError('pending_master')]
                    pending_master = [pending_id, 'master_pending_len1D']

                else:
                    raise NotImplementedError
                
        return gen1D, pending_master


    def gen_floats(gen_name, list_length, sign, bound, dist):
        gen1D = gen_ints(gen_name, list_length, sign, bound, random_func=np.random.uniform)

        return gen1D


    def batch_embody(gen_name: str, batch_size: int, this_k: int, demo: list):
        """
        Given ONE SINGLE idata sample template, and ALL locations to find those lst1D, generate a batch of lst1D to fill in.
        Returns:
            gen2D: a 2-D list, dimensions means: [batch, embodied arg]
        Explain:
            int requires:
                nums:     !fix_len:5! / !varing_len:6.5:4.21!
                positiveness:  $p0n$ / $b01$ / $p$ / $p0$
                absolute upperbound: ^10^ / ^100^ / ^1000^
                distribution type:  ~uniformly~ / ~same~ / ~around~
                    e.g.:   '.. ^1000^~same~' -> [577, 577, 577]  | around: [577, 576, 577] (± 2)
            str requires:
                nums
            float requires:
                nums:     !fix_len:5! / !varing_len:6.5:4.21!
                positiveness:  $p0n$ / $b01$ / $p$ / $p0$
                absolute upperbound: ^10^ / ^100^ / ^1000^
        """

        nums = re.findall(r"!(.*?):(.*?)!", gen_name)[0]
        gen2D = None
        pending_master = False

        if "int" in gen_name:
            if "$b01$" in gen_name:
                gen2D = [generate_binary(nums) for _ in range(batch_size)]
            else:
                sign = re.findall(r"\$(.*?)\$", gen_name)[0]
                bound = re.findall(r"\^(.*?)\^", gen_name)[0]
                gen2D = [gen_ints(gen_name, nums, sign, int(bound), this_k=this_k) for _ in range(batch_size)]
                gen2D, pending_masters = list(zip(*gen2D))
                for x in pending_masters:
                    if x:
                        pending_master = x


        elif "str" in gen_name:
            list_of_str = list(itertools.chain(*demo))
            try:
                char_choices = set(itertools.chain(*list_of_str))
            except:
                pass

            lens = [len(x) for x in list_of_str]
            if all_equal(lens):
                # str_crossover
                str2D = np.array([list(s) for s in list_of_str])
                C, L = str2D.shape
                gen2D = []
                for l in range(L):
                    gen2D.append(np.random.choice(str2D[:,l], batch_size))
                gen2D = np.asarray(gen2D).T.tolist()
                gen2D = [[''.join(x)] for x in gen2D]
            else:

                len_of_strs = [len(x) for x in list_of_str]
                med = np.median(len_of_strs)
                std = min(med*0.7, np.std(len_of_strs))
                list_length = ['varing_len', f'{med}:{std}']
                gen_lens = [decide_generated_len_of_lst1D(list_length) for _ in range(batch_size)]
                
                def random_str(l):
                    res = np.random.choice(list(char_choices), l).tolist()
                    return ''.join(res)
                gen2D = [[random_str(l)] for l in gen_lens]


        elif "float" in gen_name:
            sign = re.findall(r"\$(.*?)\$", gen_name)[0]
            bound = re.findall(r"\^(.*?)\^", gen_name)[0]
            dist = re.findall(r"~(.*?)~", gen_name)[0]
            gen2D = [gen_floats(gen_name, nums, sign, int(bound), dist) for _ in range(batch_size)]

        else:
            raise NotImplementedError

        return gen2D, pending_master


    class Filler:
        def __init__(self, target, orig_heter, orig_homo):
            self.orig_demo = orig_heter + orig_homo
            self.num_samples = len(target)
            self.genName2locs = [defaultdict(list) for _ in range(self.num_samples)] # for each sample, init a new dict
            self.target = target
        def record(self, i_samp, i_arg, value):
            self.genName2locs[i_samp][value].append(i_arg)
            return
        def locs2demo(self, locs):
            if locs[0]<agreed_pre_len:
                return self.orig_demo[locs[0]]
            else:
                i_homo = (locs[0] - agreed_pre_len) % agreed_group_len
                return self.orig_demo[agreed_pre_len + i_homo]
        def fill(self):
            pending_master = False
            for i_samp in range(self.num_samples):
                this_k = inferred_ks[i_samp]
                for _genName, locs in self.genName2locs[i_samp].items():
                    genName = _genName.split('@CNT:')[0]
                    tofill, tmp_master = batch_embody(genName, batch_size = len(locs), this_k = this_k, demo = self.locs2demo(locs))
                    if tmp_master:
                        pending_master = tmp_master
                        
                    for i, loc in enumerate(locs):
                        self.target[i_samp][loc] = tofill[i]
            if pending_master:
                pending_heter_idx, info = pending_master
                for i_samp in range(self.num_samples):
                    if info=='master_pending_len1D':
                        self.target[i_samp][agreed_master_id_meaning[0]][0] = len(self.target[i_samp][pending_heter_idx])
                    else:
                        raise NotImplementedError
            
            return self.target

    final_aug_idata_sat = copy.deepcopy(final_tmpl_sat)

    filler = Filler(
        target = final_aug_idata_sat, 
        orig_heter = agreed_datas_heter, 
        orig_homo = agreed_datas_homo)
    for i_samp in range(required_size):
        for i_arg, arg in enumerate(final_tmpl_sat[i_samp]):
            filler.record(i_samp, i_arg, final_tmpl_sat[i_samp][i_arg])

    final_aug_idata_sat = filler.fill()
    return final_aug_idata_sat


