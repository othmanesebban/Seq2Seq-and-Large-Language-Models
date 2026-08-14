import math
import operator
import sys
import json
from functools import reduce
from collections import Counter

def lcs_length(x, y):
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

def rouge_l(candidate, references):
    precisions, recalls = [], []
    for reference in references:
        lcs = lcs_length(candidate.split(), reference.split())
        ref_len = len(reference.split())
        cand_len = len(candidate.split())
        precisions.append(lcs / cand_len if cand_len > 0 else 0)
        recalls.append(lcs / ref_len if ref_len > 0 else 0)
    precision = sum(precisions) / len(references)
    recall = sum(recalls) / len(references)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1

def cider(candidate, references, n=4):
    """Calculate CIDEr score."""
    def ngram_counts(sentence, n):
        words = sentence.split()
        return Counter([' '.join(words[i:i+n]) for i in range(len(words) - n + 1)])

    def compute_tf_idf(ref_ngrams, ngram):
        # Compute term frequency (TF)
        tf = 1.0 / len(ref_ngrams) if ngram in ref_ngrams else 0

        # Compute inverse document frequency (IDF)
        doc_count = sum(ngram in ref for ref in ref_ngrams)
        idf = math.log(len(ref_ngrams) / (1 + doc_count)) if doc_count > 0 else 0
        return tf * idf

    # Compute n-gram counts for candidate and references
    cand_ngrams = [ngram_counts(candidate, i) for i in range(1, n + 1)]
    ref_ngrams = {i: [ngram_counts(ref, i) for ref in references] for i in range(1, n + 1)}

    cider_scores = []
    for i in range(1, n + 1):
        overlap = 0.0
        norm_cand = 0.0
        norm_ref = 0.0
        for ngram, count in cand_ngrams[i - 1].items():
            # CIDEr uses TF-IDF weighted cosine similarity
            tfidf_cand = compute_tf_idf(ref_ngrams[i], ngram) * count
            tfidf_refs = sum(compute_tf_idf(ref, ngram) for ref in ref_ngrams[i])
            overlap += tfidf_cand * tfidf_refs
            norm_cand += tfidf_cand ** 2
            norm_ref += tfidf_refs ** 2

        # Avoid division by zero
        norm_cand = math.sqrt(norm_cand)
        norm_ref = math.sqrt(norm_ref)
        if norm_cand > 0 and norm_ref > 0:
            cider_scores.append(overlap / (norm_cand * norm_ref))
        else:
            cider_scores.append(0)
    return sum(cider_scores) / len(cider_scores)

def count_ngram(candidate, references, n):
    clipped_count = 0
    count = 0
    r = 0
    c = 0
    for si in range(len(candidate)):
        ref_counts = []
        ref_lengths = []
        for reference in references:
            ref_sentence = reference[si]
            ngram_d = {}
            words = ref_sentence.strip().split()
            ref_lengths.append(len(words))
            limits = len(words) - n + 1
            for i in range(limits):
                ngram = ' '.join(words[i:i+n]).lower()
                if ngram in ngram_d.keys():
                    ngram_d[ngram] += 1
                else:
                    ngram_d[ngram] = 1
            ref_counts.append(ngram_d)
        cand_sentence = candidate[si]
        cand_dict = {}
        words = cand_sentence.strip().split()
        limits = len(words) - n + 1
        for i in range(0, limits):
            ngram = ' '.join(words[i:i + n]).lower()
            if ngram in cand_dict:
                cand_dict[ngram] += 1
            else:
                cand_dict[ngram] = 1
        clipped_count += clip_count(cand_dict, ref_counts)
        count += limits
        r += best_length_match(ref_lengths, len(words))
        c += len(words)
    if clipped_count == 0:
        pr = 0
    else:
        pr = float(clipped_count) / count
    bp = brevity_penalty(c, r)
    return pr, bp

def clip_count(cand_d, ref_ds):
    count = 0
    for m in cand_d.keys():
        m_w = cand_d[m]
        m_max = 0
        for ref in ref_ds:
            if m in ref:
                m_max = max(m_max, ref[m])
        m_w = min(m_w, m_max)
        count += m_w
    return count

def best_length_match(ref_l, cand_l):
    least_diff = abs(cand_l - ref_l[0])
    best = ref_l[0]
    for ref in ref_l:
        if abs(cand_l - ref) < least_diff:
            least_diff = abs(cand_l - ref)
            best = ref
    return best

def brevity_penalty(c, r):
    if c > r:
        bp = 1
    else:
        bp = math.exp(1 - (float(r) / c))
    return bp

def geometric_mean(precisions):
    return (reduce(operator.mul, precisions)) ** (1.0 / len(precisions))

def BLEU(s, t, flag=False, max_n=4):
    candidate = [s.strip()]
    if flag:
        references = [[t[i].strip()] for i in range(len(t))]
    else:
        references = [[t.strip()]]
    precisions = []
    for n in range(1, max_n + 1):
        pr, bp = count_ngram(candidate, references, n)
        precisions.append(pr)
    scores = [geometric_mean(precisions[:i]) * bp for i in range(1, max_n + 1)]
    return scores

if __name__ == "__main__":
    test = json.load(open('MLDS_hw2_1_data/testing_label.json', 'r'))
    output = sys.argv[1]
    result = {}
    with open(output, 'r') as f:
        for line in f:
            line = line.rstrip()
            comma = line.index(',')
            test_id = line[:comma]
            caption = line[comma + 1:]
            result[test_id] = caption
    bleu_scores = [0, 0, 0, 0]
    rouge_l_scores = {"precision": 0, "recall": 0, "f1": 0}
    cider_scores = []
    for item in test:
        captions = [x.rstrip('.') for x in item['caption']]
        scores = BLEU(result[item['id']], captions, True)
        for i in range(4):
            bleu_scores[i] += scores[i]
        precision, recall, f1 = rouge_l(result[item['id']], captions)
        rouge_l_scores["precision"] += precision
        rouge_l_scores["recall"] += recall
        rouge_l_scores["f1"] += f1
        cider_scores.append(cider(result[item['id']], captions))
    bleu_scores = [score / len(test) for score in bleu_scores]
    rouge_l_scores = {k: v / len(test) for k, v in rouge_l_scores.items()}
    cider_score = sum(cider_scores) / len(cider_scores)
    print(f"B-1 = {bleu_scores[0]:.6f}, B-2 = {bleu_scores[1]:.6f}, "
          f"B-3 = {bleu_scores[2]:.6f}, B-4 = {bleu_scores[3]:.6f}")
    print(f"ROUGE-L Precision = {rouge_l_scores['precision']:.6f}, "
          f"Recall = {rouge_l_scores['recall']:.6f}, F1 = {rouge_l_scores['f1']:.6f}")
    print(f"CIDEr = {cider_score:.6f}")
