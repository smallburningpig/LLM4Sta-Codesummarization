import os
import pandas as pd
import nltk
import sacrebleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import re
import xml.sax.saxutils
from bleu import bleuFromMaps
from sentence_transformers import SentenceTransformer, util
nltk.download('punkt')
nltk.download('wordnet')

def splitPuncts(line):
    return ' '.join(re.findall(r"[\w]+|[^\s\w]", line))

def euclidean_distance(x, y):
    distance = 0.0
    for i in range(len(x)):
        distance += (x[i] - y[i])**2
    return distance**0.5

def sentence_bert_score_cos(output, gold):
    model = SentenceTransformer('D:\hugging-face\\all-MiniLM-L6-v2')

    embeddings1 = model.encode(output, convert_to_tensor=True)
    embeddings2 = model.encode(gold, convert_to_tensor=True)

    cosine_scores = util.cos_sim(embeddings1, embeddings2)

    score = 0
    num = 0

    for i in range(len(output)):
        num = num + 1

        score = score + cosine_scores[i][i]

    print("Number of cases: {}".format(num))
    print("Score: {:.4f}".format(score/num))

    return score/num



def sentence_bert_score_euc(output, gold):
    model = SentenceTransformer('D:\hugging-face\\all-MiniLM-L6-v2')
    embeddings1 = model.encode(output, convert_to_tensor=True)
    embeddings2 = model.encode(gold, convert_to_tensor=True)

    score = 0
    num = 0
    for i in range(len(output)):
        euc_sim = euclidean_distance(embeddings1[i],embeddings2[i])
        num = num + 1
        score = score + euc_sim

    print("Number of cases:{}".format(num))
    print("Score: {:.4f}".format(score/num))
    return score/num


def sentence_bert_score(output, gold):
    print('**********')
    print('Sentence Bert + cosine similarity')
    scos = sentence_bert_score_cos(output, gold)
    print('---------')
    print('Sentence Bert + Euclidean distance')
    seuc = sentence_bert_score_euc(output, gold)
    print('**********')
    return scos, seuc


def bleu(output, gold):
    score = 0
    num = 0
    for i in range(len(output)):
        num = num + 1
        predictionMap = {}
        goldMap = {}
        predictionMap[i] = [splitPuncts(output[i].strip().lower())]
        goldMap[i] = [splitPuncts(gold[i].strip().lower())]
        dev_bleu = round(bleuFromMaps(goldMap, predictionMap)[0], 2)
        score += dev_bleu

    # print("Number of cases:{}".format(num))
    # print("Score: {:.4f}".format(score/num))
    return score/num

def normalize(s):
    '''Normalize and tokenize text to match the new BLEU implementation.'''
    normalize1 = [
        ('<skipped>', ''),
        (r'-\n', ''),
        (r'\n', ' '),
    ]
    normalize1 = [(re.compile(pattern), replace) for (pattern, replace) in normalize1]

    normalize2 = [
        (r'([\{-\~\[-\` -\&\(-\+\:-\@\/])', r' \1 '),
        (r'([^0-9])([\.,])', r'\1 \2 '),
        (r'([\.,])([^0-9])', r' \1 \2'),
        (r'([0-9])(-)', r'\1 \2 ')
    ]
    normalize2 = [(re.compile(pattern), replace) for (pattern, replace) in normalize2]

    if isinstance(s, list):
        s = " ".join(s)

    for (pattern, replace) in normalize1:
        s = re.sub(pattern, replace, s)
    s = xml.sax.saxutils.unescape(s, {'&quot;': '"'})
    s = " %s " % s.lower()

    for (pattern, replace) in normalize2:
        s = re.sub(pattern, replace, s)
    return s.split()


def compute_bleu(reference, candidate):
    reference = [reference] 
    bleu_score = sacrebleu.corpus_bleu([candidate], [reference]).score  # 返回的是 BLEU 分数百分比
    return bleu_score


def compute_rougeL(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    score = scorer.score(reference, candidate)
    return score['rougeL'].fmeasure  # 返回 ROUGE-L 的 F1 分数


def compute_meteor(reference, candidate):
    reference_tokenized = normalize(reference)
    candidate_tokenized = normalize(candidate)
    return meteor_score([reference_tokenized], candidate_tokenized)


def load_csv(groundtruth_file, test_file):
    groundtruth_df = pd.read_csv(groundtruth_file, header=None)
    test_df = pd.read_csv(test_file, header=None)
    references = groundtruth_df.iloc[:, 1].tolist()
    candidates = test_df.iloc[:, 1].tolist()
    return references, candidates

def evaluate_translations(references, candidates):
    total_bleu, total_rougeL, total_meteor = 0, 0, 0
    num_sentences = len(references)

    for i in range(num_sentences):
        reference = references[i].lower()  
        candidate = candidates[i].lower() 

        bleu_score = bleu([candidate], [reference])  
        total_bleu += bleu_score

        rougeL = compute_rougeL(reference, candidate)
        total_rougeL += rougeL

        meteor = compute_meteor(reference, candidate)
        total_meteor += meteor

    avg_bleu = total_bleu / num_sentences if num_sentences > 0 else 0
    avg_rougeL = total_rougeL / num_sentences if num_sentences > 0 else 0
    avg_meteor = total_meteor / num_sentences if num_sentences > 0 else 0

    return avg_bleu, avg_rougeL, avg_meteor


def evaluate_all_files(groundtruth_file, folder):
    for filename in os.listdir(folder):
        if filename.endswith('.csv'):
            test_file = os.path.join(folder, filename)

            references, candidates = load_csv(groundtruth_file, test_file)
            avg_bleu, avg_rougeL, avg_meteor = evaluate_translations(references, candidates)

            print(f"File: {filename}")
            print(f"Average BLEU: {avg_bleu:.4f}")
            print(f"Average ROUGE-L: {avg_rougeL:.4f}")
            print(f"Average METEOR: {avg_meteor:.4f}")
            print("-" * 50)


def main():
    groundtruth_file = 'groundtruth.txt'  
    processed_folder = 'processed' 

    evaluate_all_files(groundtruth_file, processed_folder)


if __name__ == '__main__':
    main()
