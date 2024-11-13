import json
import os
import random
import shutil
import csv
import argparse
import logging
import sys
import re
import pandas as pd
from tqdm import tqdm
from model import GPT, StarChat, CodeLLAMA

def read_file_to_string(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return str(e)

def remove_comments_and_docstrings(source, lang):
    def replacer(match):
        s = match.group(0)
        return " " if s.startswith('/') else s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    temp = [x for x in re.sub(pattern, replacer, source).split('\n') if x.strip() != ""]
    return '\n'.join(temp)

def extract_scores(response):
    scores = []
    cleaned_response = response.replace('*', '')

    for i in range(1, 17):
        pattern = re.search(rf'Comment {i}:\s*(\d+)', cleaned_response)
        scores.append(int(pattern.group(1)) if pattern else None)
    return scores

def generate_summaries_zero_shot(args, model, code, output_file, cnt=0):
    args.logger.info('zero-shot prompt...')
    result_rows = []

    for idx, c in tqdm(enumerate(code)):
        if idx < cnt:
            continue

        response = model.ask(input=args.basic_prompt + c)
        scores = extract_scores(response)
        result_rows.append([idx] + scores)

        print(f'Processed index: {idx}, Scores: {scores}')

    columns = ['Index'] + [f'Summary {i + 1}' for i in range(16)]
    df = pd.DataFrame(result_rows, columns=columns)
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", default="./output_cases", type=str)
    parser.add_argument("--language", default="java", type=str)
    parser.add_argument("--model", default="gpt-4o", type=str)
    parser.add_argument("--mode", default="w", type=str)
    parser.add_argument("--basic_prompt", default='''"I will provide you with a piece of function-level code, a block of statement-level code, and 16 summaries written for the statement-level code. Please analyze the statement-level code and rate each of the summaries on a scale of 1 to 5, with 5 being the highest score. Please note that good statement-level code summaries should be concise and clear, summarizing the main function and purpose of the statement-level code, and clearly expressing its core logic or design intentions. At the same time, they should avoid overly detailed implementation specifics, ensuring that other developers can quickly understand the role of the statement.Please provide your output in the format: Comment i: {your rating}.\n"''', type=str)
    parser.add_argument("--openai_key", default='', type=str)
    parser.add_argument("--max_new_tokens", default=4096, type=int)
    parser.add_argument("--temperature", default=0.1, type=float)
    parser.add_argument("--write_groundtruth", default=True, type=bool)
    parser.add_argument("--top_k", default=50, type=int)
    parser.add_argument("--top_p", default=0.5, type=float)
    parser.add_argument("--log_filename", default='log.txt', type=str)
    args = parser.parse_args()
    dir = './{}/{}/{}/'.format(args.language, args.model, args.temperature)
    if not os.path.exists(dir):
        os.makedirs(dir)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    args.logger = logging.getLogger(__name__)
    log_file_path = os.path.join(os.path.join(dir, args.log_filename))
    fh = logging.FileHandler(log_file_path)
    args.logger.addHandler(fh)
    args.logger.info("Training/evaluation parameters %s", args)
    args.logger.info("\n")
    MODEL_NAME_OR_PATH = {
        'gpt-4': 'gpt-4-1106-preview',
        'gpt-3.5': 'gpt-3.5-turbo',
        'gpt-4o': 'gpt-4o',
        'gpt-4o-mini': 'gpt-4o-mini',
        'starchat': '/home/jspi/data/mmp/starchat/starchat',
        'codellama': '/home/david/MY/codellama/CodeLlama-7b-Instruct-hf'
    }
    args.model_name_or_path = MODEL_NAME_OR_PATH[args.model]
    if args.model in ['gpt-4', 'gpt-3.5', 'gpt-4o', 'gpt-4o-mini']:
        model = GPT(args=args)
    elif args.model == 'starchat':
        model = StarChat(args=args)
    elif args.model == 'codellama':
        model = CodeLLAMA(args=args)
    else:
        print('Model not found!')
        sys.exit(1)

    txt_folder_path = args.data_folder
    txt_files = [os.path.join(txt_folder_path, file) for file in os.listdir(txt_folder_path) if file.endswith('.txt')]

    code = [remove_comments_and_docstrings(read_file_to_string(txt_file), 'java') for txt_file in txt_files]

    output_file_name = f'results.csv'
    generate_summaries_zero_shot(args, model, code=code, output_file=output_file_name, cnt=0)

if __name__ == '__main__':
    main()
