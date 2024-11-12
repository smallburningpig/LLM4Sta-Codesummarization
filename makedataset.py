import os
import json
import re


def remove_multiline_comments(code):
    cleaned_code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    return cleaned_code


def has_matching_parentheses(code):
    stack = []
    parentheses = {'(': ')', '{': '}', '[': ']'}
    for char in code:
        if char in parentheses:
            stack.append(char)
        elif char in parentheses.values():
            if not stack or parentheses[stack.pop()] != char:
                return False
    return not stack


def is_valid_comment(comment):
    if not re.match(r'^[a-zA-Z0-9\s]+$', comment):
        return False
    word_count = len(comment.split())
    return word_count >= 3


def process_jsonl_file(file_path, output_file):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]

    original_count = len(data)
    processed_count = 0

    with open(output_file, 'a', encoding='utf-8') as target_file:
        for entry in data:
            clean_code = remove_multiline_comments(entry['code'])
            lines_of_code = clean_code.splitlines()
            for index, line in enumerate(lines_of_code):
                if '//' in line and not line.strip().startswith('//'):
                    parts = line.split('//', 1)
                    code = parts[0].strip()
                    comments = parts[1].strip() if len(parts) > 1 else None
                    if comments and is_valid_comment(
                            comments) and "import" not in code and "package" not in code and code.strip().endswith(
                            ';') and re.match(r'^[a-zA-Z]', code) and has_matching_parentheses(code):
                        processed_count += 1
                        print(code)
                        result = {
                            'statement_level_code': code,
                            'statement_level_comments': comments,
                            'funtion_level_code': entry['original_string'],
                            'funtion_level_comment_docstring': entry['docstring'],
                            'repo': entry['repo'],
                            'path': entry['path'],
                            'func_name': entry['func_name'],
                            'language': entry['language'],
                            'funtion_level_code_tokens': entry['code_tokens'],
                            'funtion_level_docstring_tokens': entry['docstring_tokens'],
                            'url': entry['url'],
                            'sha': entry['sha'],
                            'partition': entry['partition']
                        }
                        # print(result)
                        json.dump(result, target_file)
                        target_file.write('\n')

    return original_count, processed_count


def merge_and_process_jsonl_files(folder_path, output_file):
    open(output_file, 'w').close()

    total_original_count = 0
    total_processed_count = 0

    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.jsonl'):
                file_path = os.path.join(root, filename)
                print(f"正在处理文件: {file_path}")
                original_count, processed_count = process_jsonl_file(file_path, output_file)
                total_original_count += original_count
                total_processed_count += processed_count

    print(f"处理前的总数据条数: {total_original_count}")
    print(f"处理后的总数据条数: {total_processed_count}")



folder_path = 'dataset'
output_file = 'merged_output.jsonl'
merge_and_process_jsonl_files(folder_path, output_file)
