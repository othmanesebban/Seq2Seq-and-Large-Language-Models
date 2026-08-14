import math
import operator
import sys
import json
from functools import reduce
from pycocoevalcap.cider.cider import Cider
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk
nltk.download('omw-1.4')
nltk.download('wordnet')
import sys
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
                ngram = ' '.join(words[i:i + n]).lower()
                if ngram in ngram_d:
                    ngram_d[ngram] += 1
                else:
                    ngram_d[ngram] = 1
            ref_counts.append(ngram_d)
        cand_sentence = candidate[si]
        cand_dict = {}
        words = cand_sentence.strip().split()
        limits = len(words) - n + 1
        for i in range(limits):
            ngram = ' '.join(words[i:i + n]).lower()
            if ngram in cand_dict:
                cand_dict[ngram] += 1
            else:
                cand_dict[ngram] = 1
        clipped_count += clip_count(cand_dict, ref_counts)
        count += limits
        r += best_length_match(ref_lengths, len(words))
        c += len(words)
    pr = float(clipped_count) / count if clipped_count > 0 else 0
    bp = brevity_penalty(c, r)
    return pr, bp

def clip_count(cand_d, ref_ds):
    count = 0
    for m in cand_d:
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
    return 1 if c > r else math.exp(1 - (float(r) / c))

def geometric_mean(precisions):
    return (reduce(operator.mul, precisions)) ** (1.0 / len(precisions))

def BLEU(candidate, references, multiple_references=False):
    score = 0.0
    candidate = [candidate.strip()]
    if multiple_references:
        references = [[ref.strip()] for ref in references]
    else:
        references = [[references.strip()]]
    precisions = []
    pr, bp = count_ngram(candidate, references, 1)
    precisions.append(pr)
    score = geometric_mean(precisions) * bp
    return score

def evaluate_cider(result, references):
    cider_scorer = Cider()
    hypotheses = {key: [result[key]] for key in result}
    formatted_references = {key: references[key] for key in result}
    scores, _ = cider_scorer.compute_score(formatted_references, hypotheses)
    return scores

def evaluate_meteor(result, references):
    meteor_scores = []
    for key in result:
        hypothesis = result[key].split()
        reference_list = [ref.split() for ref in references[key]]
        score = meteor_score(reference_list, hypothesis)
        meteor_scores.append(score)
    return sum(meteor_scores) / len(meteor_scores)

def evaluate_rouge(result, references):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = []
    for key in result:
        hypothesis = result[key]
        reference_list = references[key]
        scores = [scorer.score(ref, hypothesis)['rougeL'].fmeasure for ref in reference_list]
        rouge_scores.append(max(scores))
    return sum(rouge_scores) / len(rouge_scores)

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

    bleu_scores = []
    cider_references = {}
    meteor_references = {}
    rouge_references = {}

    for item in test:
        captions = [x.rstrip('.') for x in item['caption']]
        candidate = result[item['id']]
        bleu_scores.append(BLEU(candidate, captions, True))
        cider_references[item['id']] = captions
        meteor_references[item['id']] = captions
        rouge_references[item['id']] = captions

    bleu_average = sum(bleu_scores) / len(bleu_scores)
    cider_score = evaluate_cider(result, cider_references)
    meteor_score_avg = evaluate_meteor(result, meteor_references)
    rouge_l_score = evaluate_rouge(result, rouge_references)

    print(f"Average BLEU score: {bleu_average}")
    print(f"Average CIDEr score: {cider_score}")
    print(f"Average METEOR score: {meteor_score_avg}")
    print(f"Average ROUGE-L score: {rouge_l_score}")
