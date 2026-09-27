# CS5760 Homework 2

## 👤 Student Information

| Field | Details |
| --- | --- |
| Student Name | Srinivas Reddy Pittu |
| Student ID | 700777009 |
| University | University of Central Missouri |
| Department | Data Science & AI |
| Course | CS5760 Natural Language Processing |
| Semester | Fall 2026 |

## 📌 Overview

This homework covers Naive Bayes text classification, harms of classification, bigram language models, Laplace smoothing, backoff, and multiclass evaluation. It combines written calculations with two Python programs that demonstrate how the methods work.

The final report, [Homework 2_Pittu_700777009.docx](Homework%202_Pittu_700777009.docx), contains handwritten calculations, written explanations, commented Python code, and screenshots of the program outputs.

## 📂 Submission Files

| File | Purpose |
| --- | --- |
| `Homework 2_Pittu_700777009.docx` | Final homework report with answers, calculations, code, and output screenshots. |
| `Metrics_Bigram_HW2.py` | Python implementation for Part I, Question 5 and Python implementation for Part II, Question 1. |
| `README.md` | Assignment overview, implementation details, results, and run instructions. |

The two Python programs are also included in the Word report. The script names above match the source files used in the run instructions below.

## Part I — Written Calculations and Explanations

### Q1 — Naive Bayes Document Classification

The document **“predictable no fun”** is classified using Multinomial Naive Bayes with add-1 smoothing. The class priors are `P(negative) = 3/5` and `P(positive) = 2/5`. The vocabulary contains 20 words, with 14 tokens in the negative class and 9 tokens in the positive class.

The class scores are:

```text
Negative = (3/5) × (2/34) × (2/34) × (1/34)
         ≈ 0.0000610624873

Positive = (2/5) × (1/29) × (1/29) × (2/29)
         ≈ 0.0000328016729
```

The predicted class is **Negative** because its score is larger. These values are unnormalized class scores, calculated by multiplying the prior by the word likelihoods.

### Q2 — Harms of Classification

The written answers explain three issues:

- **Representational harm:** A classifier can reinforce negative stereotypes about a social group. The Kiritchenko and Mohammad (2018) study described in the lecture found lower sentiment scores and more negative emotion for sentences with African American names than for otherwise identical sentences with European American names.
- **Censorship:** Toxicity classifiers may incorrectly flag harmless mentions of minority identities, reducing the visibility of valid speech.
- **Performance differences:** Limited representation of African American English or Indian English in training data can cause classifiers to perform poorly on these varieties of English.

### Q3 — Bigram Probabilities and the Zero-Probability Problem

Bigram probabilities are estimated using maximum likelihood estimation (MLE):

```text
P(next word | previous word)
    = count(previous word, next word) / count(previous word with a successor)
```

The probability of each sentence includes its start and end transitions:

| Sentence | Probability Product | Result |
| --- | --- | ---: |
| S1: `<s> I love NLP </s>` | `(2/3) × 1 × (1/2) × 1` | `1/3 ≈ 0.333333` |
| S2: `<s> I love deep learning </s>` | `(2/3) × 1 × (1/2) × 1 × (1/2)` | `1/6 ≈ 0.166667` |

**S1 is twice as probable as S2.**

The bigram `ate noodle` is unseen, so its MLE probability is `0/12 = 0`. A zero transition makes the entire sentence probability zero and leads to infinite perplexity. With add-1 smoothing, vocabulary size 10, and total count after `ate` equal to 12:

```text
P(noodle | ate) = (0 + 1) / (12 + 10)
               = 1/22
               ≈ 0.045455
```

This calculation treats `noodle` as part of the stated vocabulary.

### Q4 — Trigram to Bigram Backoff

The training corpus contains `I like cats`, `I like dogs`, and `You like cats`.

```text
P(cats | I, like)
    = count(I like cats) / count(I like)
    = 1/2
    = 0.5
```

The trigram `You like dogs` is absent. Backing off to the shorter context gives:

```text
P(dogs | like)
    = count(like dogs) / (count(like cats) + count(like dogs))
    = 1 / (2 + 1)
    = 1/3
    ≈ 0.333333
```

Backoff uses an observed shorter context when the longer context has insufficient data. This exercise uses the bigram probability directly because no discount or backoff weight is specified.

### Q5 — Multiclass Evaluation

The confusion matrix contains 90 animals. **Rows are system predictions, and columns are gold labels.**

| System / Gold | Cat | Dog | Rabbit |
| --- | ---: | ---: | ---: |
| Cat | 5 | 10 | 5 |
| Dog | 15 | 20 | 10 |
| Rabbit | 0 | 15 | 10 |

For each class, the diagonal value is TP. FP is the rest of the prediction row, and FN is the rest of the gold-label column.

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
```

| Class | TP | FP | FN | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cat | 5 | 15 | 15 | 0.250000 | 0.250000 |
| Dog | 20 | 25 | 25 | 0.444444 | 0.444444 |
| Rabbit | 10 | 15 | 15 | 0.400000 | 0.400000 |

| Averaging Method | Precision | Recall |
| --- | ---: | ---: |
| Macro | 0.364815 | 0.364815 |
| Micro | 0.388889 | 0.388889 |

Macro averaging calculates the metric for each class and takes the simple average, giving all classes equal weight. Micro averaging pools TP, FP, and FN before calculating the metric, giving each animal equal weight. Here, total TP is 35, total FP is 55, and total FN is 55.

The `calculate_metrics(matrix, labels)` function accepts a square matrix of nonnegative integer counts and one label per class. It returns the per-class results and the macro and micro averages. If a denominator is zero, the corresponding metric is reported as `0.0`.

## Part II — Bigram Language Model

The program uses this training corpus:

```text
<s> I love NLP </s>
<s> I love deep learning </s>
<s> deep learning is fun </s>
```

The implementation performs the following steps:

1. Splits each sentence into tokens and ensures it has one start marker and one end marker.
2. Counts unigrams and adjacent bigrams using `collections.Counter`.
3. Counts contexts with a following token and estimates bigram probabilities using MLE.
4. Multiplies transition probabilities to score a supplied sentence, including the final transition to `</s>`.
5. Prints the counts, probabilities, test results, and preferred sentence.

The corpus contains **17 tokens including boundary markers** and **14 bigram occurrences**. Bigrams are counted within each sentence, so no artificial transition is created between two sentences.

The program preserves capitalization and splits on whitespace. Sentences can be supplied with or without boundary markers. Unseen transitions receive zero because the programming task uses unsmoothed MLE. The `fractions.Fraction` class keeps the probability products exact.

### Expected Sentence Results

```text
Sentence: <s> I love NLP </s>
  Factors: 2/3 * 1 * 1/2 * 1
  Probability: 1/3 = 0.333333

Sentence: <s> I love deep learning </s>
  Factors: 2/3 * 1 * 1/2 * 1 * 1/2
  Probability: 1/6 = 0.166667

Preferred sentence: <s> I love NLP </s>
```

Both alternatives after `love` have probability `1/2`. However, ending after `learning` adds another factor of `1/2`, while ending after `NLP` has probability `1`. This makes S1 twice as probable as S2.

## How to Run

### Requirements

- Python 3.
- No third-party packages or external datasets are required.
- The bigram program uses only `collections` and `fractions` from the Python standard library.

### Run Locally

Open a terminal in the folder containing the Python files and run:

```bash
python Metrics_Bigram_HW2.ipynb
```

Use `python3` instead of `python` if that is the Python command on your system. Both scripts include their input data and run without additional arguments.

### Run in Google Colab

1. Open a Python notebook in Google Colab.
2. Upload the `.ipynb` files through the notebook's Files panel.
3. Run the following commands in separate cells:

```python
%run Metrics_Bigram_HW2.ipynb
```

The first program prints the evaluation metrics. The second prints unigram counts, bigram counts, MLE probabilities, and the sentence comparison. The expected outputs are also shown in the final Word report.

