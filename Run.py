import csv
import argparse
import json
import logging
import os
import re
import sys
from io import StringIO
import tokenize
from tqdm import tqdm
import requests
from time import sleep
import openai

class GPT:
    def __init__(self, args):
        self.openai_key = ''
        self.model_name = 'gpt-4o-mini'
        self.logger = args.logger
        self.temperature = args.temperature
        self.top_p = args.top_p

    def ask(self, input, history=[], system_prompt="DEFAULT_SYSTEM_PROMPT"):
        openai.api_key = self.openai_key
        message = [{"role": "system", "content": system_prompt}]
        for his in history:
            q, a = his
            message.append({"role": "user", "content": q})
            message.append({"role": "assistant", "content": a})
        message.append({"role": "user", "content": input})
        self.logger.info('message:')
        self.logger.info(message)
        response = openai.ChatCompletion.create(model=self.model_name, messages=message, temperature=self.temperature, top_p=self.top_p)
        result = response['choices'][0]['message']['content']
        self.logger.info('result:')
        self.logger.info(result)
        sleep(1)
        return result.strip()

class GPTASK:
    def __init__(self, args):
        self.gpt = GPT(args)

    def ask(self, prompt, history=[]):
        return self.gpt.ask(prompt, history=history)

def generate_summaries_zero_shot(args, model, code, output_file, cnt=0):
    args.logger.info('zero-shot prompt with zeroshot')
    with open(output_file, args.mode, encoding="utf-8") as f:
        writer = csv.writer(f)
        for idx, c in tqdm(enumerate(code)):
            history = []
            if idx < cnt:
                continue
            try:
                message = model.ask(c, history)
            except Exception as e:
                args.logger.error(f"Error processing code at index {idx}: {e}")
                message = ""
            print(message)
            writer.writerow([idx, message])
            print(f'current idx: {idx}')

def generate_summaries_few_shot(args, model, code, output_file, cnt=0):
    args.logger.info('few-shot prompt with few-shot examples')
    with open(output_file, args.mode, encoding="utf-8") as f:
        writer = csv.writer(f)
        few_shot_history = [
            ('''Function Level Code:
def get_user_info(user_id):
    user = db.get_user(user_id)
    return user
Please understand the code above and generate a concise comment in one sentence for the following code:
user = db.get_user(user_id)''', "Retrieve the user object from the database using the user ID."),
            ('''Function Level Code:
def calculate_discount(price, discount_rate):
    discount = price * discount_rate
    return discount
Please understand the code above and generate a concise comment in one sentence for the following code:
discount = price * discount_rate''', "Calculate the discount by multiplying the price with the discount rate."),
            ('''Function Level Code:
def log_message(message):
    log_file.write(message)
    return
Please understand the code above and generate a concise comment in one sentence for the following code:
log_file.write(message)''', "Write the given message to the log file."),
            ('''Function Level Code:
def check_permission(user):
    if user.is_admin:
        return True
    return False
Please understand the code above and generate a concise comment in one sentence for the following code:
if user.is_admin:''', "Check if the user has admin privileges.")
        ]
        for idx, c in tqdm(enumerate(code)):
            if idx < cnt:
                continue
            try:
                history = few_shot_history.copy()
                history.append((f"{c}", ""))
                message = model.ask(c, history)
            except Exception as e:
                args.logger.error(f"Error processing code at index {idx}: {e}")
                message = ""
            print(message)
            writer.writerow([idx, message])
            print(f'current idx: {idx}')

def generate_summaries_chain_of_thought(args, model, funtion_level_codes, statement_level_codes, output_file, cnt=0):
    args.logger.info('Generating summaries with chain-of-thought...')
    with open(output_file, args.mode, encoding="utf-8") as f:
        writer = csv.writer(f)
        for idx in tqdm(range(len(funtion_level_codes))):
            if idx < cnt:
                continue
            try:
                history = []
                funtion_level_code = funtion_level_codes[idx]
                statement_level_code = statement_level_codes[idx]
                function_understanding = model.ask('What is the function of this code? ' + funtion_level_code, history.copy())
                history.append(('What is the function of this code? ' + funtion_level_code, function_understanding))
                statement_analysis = model.ask(f'What is the role of {statement_level_code} in the above code?', history.copy())
                history.append((f'What is the role of {statement_level_code} in the above code?', statement_analysis))
                final_summary = model.ask(f'Based on the analysis above, please use one sentence to generate a concise comment in imperative form for the following code: {statement_level_code}', history.copy())
                print(f'Function Analysis: {function_understanding}')
                print(f'Statement Analysis: {statement_analysis}')
                print(f'Final Summary: {final_summary}')
                writer.writerow([idx, final_summary])
            except Exception as e:
                args.logger.error(f"Error processing code at index {idx}: {e}")
                writer.writerow([idx, ""])

def generate_summaries_reflective(args, model, funtion_level_codes, statement_level_codes, output_file, cnt=0):
    args.logger.info('Generating summaries with reflective...')
    with open(output_file, args.mode, encoding="utf-8") as f:
        writer = csv.writer(f)
        for idx in tqdm(range(len(funtion_level_codes))):
            if idx < cnt:
                continue
            try:
                history = []
                funtion_level_code = funtion_level_codes[idx]
                statement_level_code = statement_level_codes[idx]
                first_understanding = model.ask(f"{funtion_level_code}Please use one sentence to generate a concise comment in imperative form for the following code: {statement_level_code}", history.copy())
                history.append((f"{funtion_level_code}Please use one sentence to generate a concise comment in imperative form for the following code: {statement_level_code}", first_understanding))
                self_analysis = model.ask(f"Review your previous answer and find problems with your answer.", history.copy())
                history.append(("Review your previous answer and find problems with your answer.", self_analysis))
                final_summary = model.ask(f"Based on the analysis above, please use one sentence to generate a concise comment in imperative form for the following code: {statement_level_code}", history.copy())
                print(f'First Analysis: {first_understanding}')
                print(f'Self Analysis: {self_analysis}')
                print(f'Final Summary: {final_summary}')
                writer.writerow([idx, final_summary])
            except Exception as e:
                args.logger.error(f"Error processing code at index {idx}: {e}")
                writer.writerow([idx, ""])

def remove_comments_and_docstrings(source, lang):
    if lang in ['python']:
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            if token_type == tokenize.COMMENT:
                pass
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp = [x for x in out.split('\n') if x.strip() != ""]
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            return " " if s.startswith('/') else s
        pattern = re.compile(r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"', re.DOTALL | re.MULTILINE)
        temp = [x for x in re.sub(pattern, replacer, source).split('\n') if x.strip() != ""]
        return '\n'.join(temp)

def main():
    prompt5 = """Please use one sentence to generate a concise comment in imperative form for the following code:"""
    prompt4 = """Please use one sentence to generate a concise comment in imperative form for the following code:"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default="", type=str)
    parser.add_argument("--language", default="java", type=str)
    parser.add_argument("--model", default="gpt-4omini", type=str)
    parser.add_argument("--mode", default="w", type=str)
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

    gpt_model = GPTASK(args)

    jsonl_file_path = '200case.jsonl'
    code = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            module_data = json.loads(line)
            funtion_level_code = remove_comments_and_docstrings(module_data.get('funtion_level_code', ''), 'java')
            statement_level_code = module_data.get('statement_level_code', '')
            combined_code = f"{prompt4}\n{statement_level_code}"
            code.append(combined_code)

    output_file_name = 'zero_shot.csv'
    generate_summaries_zero_shot(args, gpt_model, code, output_file_name, 0)

    jsonl_file_path = '200case.jsonl'
    code = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            module_data = json.loads(line)
            funtion_level_code = remove_comments_and_docstrings(module_data.get('funtion_level_code', ''), 'java')
            statement_level_code = module_data.get('statement_level_code', '')
            combined_code = f"Funtion Level Code:\n{funtion_level_code}\n+{prompt5}\n{statement_level_code}"
            code.append(combined_code)

    output_file_name = 'zero_shot_context.csv'
    generate_summaries_zero_shot(args, gpt_model, code, output_file_name, 0)

    funtion_level_codes = []
    statement_level_codes = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            module_data = json.loads(line)
            funtion_level_code = remove_comments_and_docstrings(module_data.get('funtion_level_code', ''), 'java')
            statement_level_code = module_data.get('statement_level_code', '')
            funtion_level_codes.append(funtion_level_code)
            statement_level_codes.append(statement_level_code)

    output_file_name = 'chainofthought.csv'
    generate_summaries_chain_of_thought(args, gpt_model, funtion_level_codes, statement_level_codes, output_file_name, 0)

    jsonl_file_path = '200case.jsonl'
    code = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            module_data = json.loads(line)
            funtion_level_code = remove_comments_and_docstrings(module_data.get('funtion_level_code', ''), 'java')
            statement_level_code = module_data.get('statement_level_code', '')
            combined_code = f"Funtion Level Code:\n{funtion_level_code}\n+{prompt5}\n{statement_level_code}"
            code.append(combined_code)
    output_file_name = 'fewshot.csv'
    generate_summaries_few_shot(args, gpt_model, code, output_file_name, 0)

    jsonl_file_path = '200case.jsonl'
    funtion_level_codes = []
    statement_level_codes = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            module_data = json.loads(line)
            funtion_level_code = remove_comments_and_docstrings(module_data.get('funtion_level_code', ''), 'java')
            statement_level_code = module_data.get('statement_level_code', '')
            funtion_level_codes.append(funtion_level_code)
            statement_level_codes.append(statement_level_code)

    output_file_name = '../result/reflective.csv'
    generate_summaries_reflective(args, gpt_model, funtion_level_codes, statement_level_codes, output_file_name,0)

if __name__ == '__main__':
    main()
