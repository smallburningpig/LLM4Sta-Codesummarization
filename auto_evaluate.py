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
# 下载 nltk 的必要资源
nltk.download('punkt')
nltk.download('wordnet')

def splitPuncts(line):
    return ' '.join(re.findall(r"[\w]+|[^\s\w]", line))
# 综合起来，该函数的目的是将输入的文本行进行处理，将其中的单词和标点符号分隔开，并用空格连接成一个新的字符串。
# 这在某些自然语言处理任务中可能是有用的，例如在文本处理或分词阶段。函数使用正则表达式来定义字词边界，确保单词和标点符号被正确地分离。

def euclidean_distance(x, y):
    distance = 0.0
    for i in range(len(x)):
        distance += (x[i] - y[i])**2
    return distance**0.5
# 综合来看，euclidean_distance 函数用于测量两个向量之间的欧氏距离，该距离表示向量在多维空间中的实际距离。在机器学习和数据分析中，欧氏距离常用于衡量数据点之间的相似性或差异性


def sentence_bert_score_cos(output, gold):
    # 加载 Sentence-BERT 模型
    model = SentenceTransformer('D:\hugging-face\\all-MiniLM-L6-v2')

    # 使用模型将生成文本和参考文本转换为嵌入向量
    embeddings1 = model.encode(output, convert_to_tensor=True)
    embeddings2 = model.encode(gold, convert_to_tensor=True)

    # 计算嵌入向量之间的余弦相似度
    cosine_scores = util.cos_sim(embeddings1, embeddings2)

    # 初始化分数和案例数
    score = 0
    num = 0

    # 遍历每个案例
    for i in range(len(output)):
        num = num + 1
        # 累加余弦相似度分数
        score = score + cosine_scores[i][i]

    # 输出案例数和平均分数
    print("Number of cases: {}".format(num))
    print("Score: {:.4f}".format(score/num))

    # 返回平均分数
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

# 分词和标点处理函数
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


# 计算 BLEU 分数（使用 SacreBLEU 库）
def compute_bleu(reference, candidate):
    reference = [reference]  # SacreBLEU 接受多个参考句子列表
    bleu_score = sacrebleu.corpus_bleu([candidate], [reference]).score  # 返回的是 BLEU 分数百分比
    return bleu_score


# 计算 ROUGE-L 分数
def compute_rougeL(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    score = scorer.score(reference, candidate)
    return score['rougeL'].fmeasure  # 返回 ROUGE-L 的 F1 分数


# 计算 METEOR 分数
def compute_meteor(reference, candidate):
    reference_tokenized = normalize(reference)
    candidate_tokenized = normalize(candidate)
    return meteor_score([reference_tokenized], candidate_tokenized)


# 读取 CSV 文件并获取第二列数据
def load_csv(groundtruth_file, test_file):
    groundtruth_df = pd.read_csv(groundtruth_file, header=None)
    test_df = pd.read_csv(test_file, header=None)
    references = groundtruth_df.iloc[:, 1].tolist()
    candidates = test_df.iloc[:, 1].tolist()
    return references, candidates


# 评估每个句子并计算总的平均分
def evaluate_translations(references, candidates):
    total_bleu, total_rougeL, total_meteor = 0, 0, 0
    num_sentences = len(references)

    for i in range(num_sentences):
        reference = references[i].lower()  # 将参考句子转换为小写
        candidate = candidates[i].lower()  # 将生成句子转换为小写

        # 计算自定义 BLEU
        bleu_score = bleu([candidate], [reference])  # 这里调用自定义的 bleu 函数
        total_bleu += bleu_score

        # 计算 ROUGE-L
        rougeL = compute_rougeL(reference, candidate)
        total_rougeL += rougeL

        # 计算 METEOR
        meteor = compute_meteor(reference, candidate)
        total_meteor += meteor

    # 计算平均分
    avg_bleu = total_bleu / num_sentences if num_sentences > 0 else 0
    avg_rougeL = total_rougeL / num_sentences if num_sentences > 0 else 0
    avg_meteor = total_meteor / num_sentences if num_sentences > 0 else 0

    return avg_bleu, avg_rougeL, avg_meteor


# 遍历处理 processed 文件夹中的所有 CSV 文件
def evaluate_all_files(groundtruth_file, folder):
    for filename in os.listdir(folder):
        if filename.endswith('.csv'):
            test_file = os.path.join(folder, filename)

            # 读取参考翻译和机器翻译
            references, candidates = load_csv(groundtruth_file, test_file)

            # 评估翻译并获取平均分数
            avg_bleu, avg_rougeL, avg_meteor = evaluate_translations(references, candidates)

            # 打印文件的最终平均分
            print(f"File: {filename}")
            print(f"Average BLEU: {avg_bleu:.4f}")
            print(f"Average ROUGE-L: {avg_rougeL:.4f}")
            print(f"Average METEOR: {avg_meteor:.4f}")
            print("-" * 50)


# 主函数
def main():
    groundtruth_file = 'groundtruth.txt'  # 参考翻译文件路径
    processed_folder = 'processed'  # 存放处理后的 CSV 文件的文件夹

    # 评估文件夹中的所有 CSV 文件
    evaluate_all_files(groundtruth_file, processed_folder)


# 运行主函数
if __name__ == '__main__':
    main()
