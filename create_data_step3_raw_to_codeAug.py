import os
import re
import time
from argparse import ArgumentParser
from difflib import SequenceMatcher
from typing import List

import numpy as np
import openai
from joblib import Parallel, delayed
from tqdm import tqdm  # noqa


from dataloaders.loader_utils import save_raw, load_all_instances, shuffled



all_keys = [
    'your_api_keys',
]

TEMPERATURES = [0.5]


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def take_atleast_one_second(func):
    """
    Custom decorator function which makes sure func takes a min of 1 second.
    :param func: Function to be decorated.
    :return: Wrapped fucntion.
    """

    def wrapper(*args, **kwargs):
        tick = time.time()
        val = func(*args, **kwargs)
        while time.time() - tick < 1:
            continue
        return val

    return wrapper


def summarizer(args, long_str: str) -> List[str]:
    """
    Summarizes the description.
    :param long_str: What you want to summarize.
    :return: Different summarized versions of the same long str.
    """
    max_tokens = max(128, int(len(long_str) * 0.1))

    @take_atleast_one_second
    def prompt_gpt3(prompt, temp):
        openai.api_key = openai.api_key = all_keys[args.which_key_id]
        output = openai.Completion.create(
            model="text-davinci-002",
            prompt=prompt.strip() + "\n\nTl;dr: ",
            temperature=temp,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return output.choices[0].text

    responses = []
    for each_temp in TEMPERATURES:
        responses.append(prompt_gpt3(long_str, each_temp))

    return responses


def predict_code_one_shot(
    args, demo_prompt: str, demo_output: str, target_prompt: str, target_starter_code: str
) -> List[str]:
    """
    Given a prompt of demo input output, predict new code for a descrption.
    :param demo_prompt: Example input.
    :param demo_output: Example output code.
    :param target_prompt: Actual question.
    :param target_starter_code: Starting seed of answer.
    :return: A list of possible solutions.
    """
    summarized_demo_prompts = summarizer(args, demo_prompt)
    summarized_target_prompts = summarizer(args, target_prompt)

    @take_atleast_one_second
    def prompt_gpt3(_demo_prompt, _demo_output, _target_prompt, _target_starter_code, _temp):
        prompt = (
            "---Question---\n"
            + _demo_prompt.strip()
            + "\n---Python Code---\n"
            + _demo_output.strip()
            + "\n---Question---\n"
            + _target_prompt.strip()
            + "\n---Python Code---\n"
            + _target_starter_code.strip()
            + "\n"
        )
        openai.api_key = all_keys[args.which_key_id]
        output = None
        while output is None:
            try:
                output = openai.Completion.create(
                    model="text-davinci-002",
                    prompt=prompt,
                    temperature=_temp,
                    max_tokens=256,
                    stop=["---"],
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
            except: 
                time.sleep(10)

        return _target_starter_code.strip() + output.choices[0].text.strip()
    
    
    all_responses = [prompt_gpt3(demo_prompt, demo_output, target_prompt, target_starter_code, temp)
        for demo_prompt in summarized_demo_prompts
        for target_prompt in summarized_target_prompts
        for temp in TEMPERATURES
    ]


    return all_responses


def parse_args():
    parser = ArgumentParser()


    num_demo_prompts = 5
    num_random_starter_prompts = 5

    machine_name = 'some_name'

    parser.add_argument("pickle_dir")
    parser.add_argument("raw_data_dir")
    parser.add_argument("code_augmented_dir")


    parser.add_argument("--num_demo_prompts", default=num_demo_prompts, type=int)
    parser.add_argument("--num_random_starter_prompts", default=num_random_starter_prompts, type=int)
    parser.add_argument("--temperatures", default=["0.2", "0.7"], nargs="+", help="Provide space separated inputs")
    parser.add_argument("--machine_name", default=machine_name)
    parser.add_argument("--which_key_id", type=int, default=0)


    parser.add_argument("--verbose", type=int, default=1)


    args = parser.parse_args()
    os.makedirs(args.code_augmented_dir, exist_ok=True)
    os.makedirs(args.pickle_dir, exist_ok=True)
    args.machine_name += f'key_{str(args.which_key_id)}'



    assert len(os.listdir(args.raw_data_dir))==4 and 'difficulty_introductory' in os.listdir(args.raw_data_dir)

    return args


def main():
    args = parse_args()
    global TEMPERATURES
    TEMPERATURES = [float(i) for i in args.temperatures]



    subfolders = [
        "difficulty_introductory",
        "difficulty_interview",
        'difficulty_competition',
        "difficulty_dm_code_contest",
    ]

    for subfolder in shuffled(subfolders):
        subfolder_dir = os.path.join(args.raw_data_dir, subfolder)
        output_dir = os.path.join(args.code_augmented_dir, subfolder)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Ready to generate to {output_dir}")


        codes_raw, codes_nameReplaced, xs_raw, ys_raw, iodatas_obj, descriptions, file_names = load_all_instances(
            subfolder_dir
        )
        print(f"Found {len(codes_raw)} instances in {subfolder_dir}")

        # iterate over each code
        for code, code_replaced, x, y, iodata, desc, file_name in tqdm(shuffled(zip(
            codes_raw, codes_nameReplaced, xs_raw, ys_raw, iodatas_obj, descriptions, file_names
        ))):
            augmented_codes = []

            # we use different demo prompt every time randomly
            for _ in range(args.num_demo_prompts):
                # index of demo prompt
                random_idx_1 = np.random.randint(0, len(codes_raw))
                random_code, random_desc = codes_raw[random_idx_1], descriptions[random_idx_1]
                # # each demo prompt has multiple codes, so we choose one of them at random
                random_idx_2 = np.random.randint(0, len(random_code))
                random_code = random_code[random_idx_2]

                # generate starter codes based on 5 random codes of target prompt
                seen_starter_codes = []
            
                for each_code in np.random.choice(code, min(len(code), args.num_random_starter_prompts), replace=False):
                    # all code till the last occurrence of "input"
                    last_line_containing_input = max(
                        i
                        for i, x in enumerate(each_line.find("input") > -1 for each_line in each_code.split("\n"))
                        if x
                    )
                    last_line_containing_input = min(
                        last_line_containing_input, int(len(each_code.split("\n")) * 0.3)
                    )  # at max keep 30% of code as starter
                    starter_code = "\n".join(each_code.split("\n")[: last_line_containing_input + 1])

                    # if generated starter code is very similar to existing starter codes, then we ignore it
                    similarity_score = max(
                        [similar(each_seen_starter_code, starter_code) for each_seen_starter_code in seen_starter_codes]
                        + [0]
                    )
                    seen_starter_codes.append(starter_code)
                    if similarity_score > 0.6:
                        continue

                    # print('\n predict new codes using GPT3 !!')
                    augmented_codes += predict_code_one_shot(args, random_desc, random_code, desc, starter_code)

            # save new codes to disk
            instance_id = int(
                re.findall(r"codes_raw_0+(\d+).py", file_name)[0]
            )  # some regex trick to parse out instance name
            save_raw(output_dir, args.machine_name, instance_id, 
                augmented_codes, '', '', '',
                x, y, iodatas_obj, '',
                '')


if __name__ == "__main__":
    main()
