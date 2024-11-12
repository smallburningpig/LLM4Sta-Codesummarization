
# LLM4Sta-CodeSummarization

This repository contains code, datasets, and evaluation scripts for the paper **"Analyzing the Performance of Large Language Models on Statement-Level Code Summarization"**. This study examines the effectiveness of large language models (LLMs) in summarizing code at the statement level, focusing on the impact of context and different prompting techniques.

## Repository Structure

- **Cat-for-statement-level-code/**: Contains tools for filtering and cleaning statement-level code comments to enhance dataset quality.
- **Dataset.jsonl**: The curated dataset used in this study, consisting of statement-level code snippets along with comments and contextual information.
- **LLMevl.py**: Evaluation script using large language models to assess code summarization quality.
- **Run.py**: Main script for running the experiments described in the paper.
- **auto_evaluate.py**: Script for automated evaluation of generated summaries using BLEU, METEOR, and ROUGE metrics.
- **auto_resultclean.py**: Tool for cleaning and organizing evaluation results.
- **makedataset.py**: Script for generating the final dataset from raw code data and comments, including context information.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Required Python packages (see `requirements.txt`)

To install the dependencies, run:
```bash
pip install -r requirements.txt
```

### Usage

1. **Data Preparation**: Run `makedataset.py` to generate the dataset for statement-level summarization tasks. The resulting dataset will include contextual information necessary for improving LLM performance.

2. **Code Summarization**: Use `Run.py` to perform the code summarization experiments with various prompting techniques (Zero-shot, Few-shot, Chain-of-Thought, and Critique).

3. **Evaluation**:
   - Run `auto_evaluate.py` to evaluate summarization quality with metrics such as BLEU, METEOR, and ROUGE.
   - Use `LLMevl.py` for LLM-based evaluations and comparisons with human assessments.
   - Clean and organize results with `auto_resultclean.py`.

### Example Commands

Generate the dataset:
```bash
python makedataset.py
```

Run the main experiments:
```bash
python Run.py
```

Evaluate results:
```bash
python auto_evaluate.py
python LLMevl.py
```

## Dataset

The dataset in `Dataset.jsonl` is structured as follows:
- **statement_level_code**: Code at the statement level.
- **statement_level_comments**: Comments for the statement-level code.
- **function_level_code**: Complete function code surrounding the statement-level code.
- **function_level_comments**: Comments for the function-level code.
- **repo**, **path**, **func_name**, **language**, **tokens**, etc.: Metadata for the code snippets.

## Results

Our experiments show that:
- **Contextual information** significantly enhances code summarization quality.
- **Few-shot prompting** is the most stable and effective prompting technique for this task.
- LLM-based evaluations demonstrate moderate agreement with human evaluations, indicating potential for automated assessments.

## License

This project is licensed under the MIT License.
