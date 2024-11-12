import os
import pandas as pd
import re

input_folder = 'result/'
output_folder = 'processed/'

def extract_and_clean_comment(text):
    text = re.sub(r'\(.*?\)', '', text)
    if '//' in text:
        first_sentence = text.split('//', 1)[1].split('.')[0] + '.'
    elif ':' in text:
        first_sentence = text.split(':', 1)[1].split('.')[0] + '.'
    else:
        text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
        return text
    cleaned_sentence = re.sub(r'[^A-Za-z0-9 ]+', '', first_sentence)
    return cleaned_sentence.strip()

for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_folder, filename)
        df = pd.read_csv(file_path, header=None)
        df.iloc[:, 1] = df.iloc[:, 1].astype(str).apply(extract_and_clean_comment)
        output_file_name = 'processed_' + filename
        output_file_path = os.path.join(output_folder, output_file_name)
        df.to_csv(output_file_path, index=False, header=False)