from noise_detection import *
from tqdm import tqdm
from bs4 import BeautifulSoup
import random
import re

class RuleCleaner(object):
    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.cleaned_data = []
        self.removed_duplicates_count = 0
        self.noisy_data = {'ContentTamper': [], 'NonLiteral': [], 'Interrogation': [], 'UnderDevelop': [],
                           'EmptyFunc': [], 'CommentOut': [], 'BlockComment': [], 'AutoCode': [], 'DuplicatedCode': []}

    def get_clean_data(self):
        for data in tqdm(self.raw_data):
            raw_code = data['statement_level_code']
            raw_comment = data['statement_level_comments']

            firstSentence = getFirstSentence(raw_comment)
            updated_comment = self.update_ContentTamper(firstSentence)
            if updated_comment != firstSentence:
                self.noisy_data['ContentTamper'].append(data)
            if if_ContentTamper(updated_comment):
                self.noisy_data['ContentTamper'].append(data)
                continue
            if if_NonLiteral(updated_comment):
                self.noisy_data['NonLiteral'].append(data)
                continue
            if if_Interrogation(updated_comment):
                self.noisy_data['Interrogation'].append(data)
                continue
            if if_UnderDevelop(updated_comment):
                self.noisy_data['UnderDevelop'].append(data)
                continue
            if if_AutoCode_by_comment(updated_comment, raw_comment):
                self.noisy_data['AutoCode'].append(data)
                continue
            if if_CommentedOut(raw_code):
                self.noisy_data['CommentOut'].append(data)
                continue
            updated_code = self.update_BlockComment(raw_code)
            if updated_code != raw_code:
                self.noisy_data['BlockComment'].append(data)
            if if_AutoCode_by_code(updated_code):
                self.noisy_data['AutoCode'].append(data)
                continue
            if if_EmptyFunc(updated_code):
                self.noisy_data['EmptyFunc'].append(data)
                continue

            cleaned_entry = data.copy()
            cleaned_entry['statement_level_code'] = updated_code
            cleaned_entry['statement_level_comments'] = updated_comment
            self.cleaned_data.append(cleaned_entry)

        self.cleaned_data = self.deduplicate_data(self.cleaned_data)
        return self.cleaned_data

    def deduplicate_data(self, data_list):
        seen = set()
        unique_data = []
        for data in data_list:
            code_comment_pair = (data['statement_level_code'], data['statement_level_comments'])
            if code_comment_pair not in seen:
                seen.add(code_comment_pair)
                unique_data.append(data)
            else:
                self.removed_duplicates_count += 1
        return unique_data

    def get_noisy_data(self):
        return self.noisy_data

    def update_BlockComment(self, raw_code):
        p = re.compile('^(\s+//)|(//)')
        new_list = []
        for row in raw_code.split('\n'):
            if not p.search(row):
                new_list.append(row)
        return '\n'.join(new_list)

    def update_ContentTamper(self, comment):
        return BeautifulSoup(comment, "html.parser").get_text()


if __name__ == '__main__':
    import json

    with open('../../merged_output.jsonl', 'r') as f:
        data_lines = f.readlines()

    raw_data = [json.loads(line.strip()) for line in data_lines]

    cleaner = RuleCleaner(raw_data)
    cleaned_data = cleaner.get_clean_data()

    print(len(cleaned_data))
    for key in cleaner.get_noisy_data():
        print(f"{key}: {len(cleaner.get_noisy_data()[key])}")

    print(f"Number of removed duplicate entries: {cleaner.removed_duplicates_count}")

    with open('cleaned_output.jsonl', 'w') as f:
        for entry in cleaned_data:
            f.write(json.dumps(entry) + '\n')

    print("Cleaned data has been saved to 'cleaned_output.jsonl'")
