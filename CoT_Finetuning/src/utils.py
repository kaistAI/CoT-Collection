import string
from rouge import Rouge
from collections import Counter

def clean_up(text):
    text =text.replace('<pad>', '')
    text = text.replace('</s>', '')
    text = text.replace(".", '')
    text = text.replace(',', '')
    text = text.replace("'", '')
    text = text.replace('"', '')
    return text   

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    
    return (white_space_fix(remove_punc(lower(s))))

def accuracy_match_score_normalize(prediction, ground_truth):
    if normalize_answer(prediction)== '' or normalize_answer(ground_truth)== '':
        return 0
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))
def exact_match_score(prediction, ground_truth):
    return accuracy_match_score_normalize(prediction, ground_truth)
def accuracy_match_score(prediction, ground_truth):
    return int(prediction.strip() == ground_truth.strip())

def calculate_rouge_scores(predictions, ground_truths):
    rouge_score = 0 
    for i in range(len(predictions)):
        prediction = predictions[i]
        ground_truth = ground_truths[i]
        rouge_score += _rougel_score(prediction, ground_truth)

    rouge_score /= len(predictions)
    return rouge_score*100

def _rougel_score(prediction, ground_truth):
    rouge = Rouge()
    # no normalization
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-l"]["f"]



def _f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def calculate_accuracy_scores(predictions, ground_truths):
    accuracy = 0
    
    for i in range(len(predictions)):
        ground_truth = ground_truths[i]
        prediction = predictions[i]
        accuracy += accuracy_match_score(prediction, ground_truth)
    
    accuracy /= len(predictions)
    return accuracy*100

def calculate_em_scores(predictions, ground_truths):
    em = 0
    
    for i in range(len(predictions)):
        ground_truth = ground_truths[i]
        prediction = predictions[i]
        em += exact_match_score(prediction, ground_truth)
    
    em /= len(predictions)
    return em*100

def calculate_f1_scores(predictions, ground_truths, ids=None):
    f1_score = 0 
    for i in range(len(predictions)):
        ground_truth = ground_truths[i]
        prediction = predictions[i]
        f1_score += _f1_score(prediction, ground_truth)

    f1_score /= len(predictions)
    return f1_score*100

def ids_to_clean_text(tokenizer, generated_ids):
    gen_text = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    
    return lmap(str.strip, gen_text)

def remain_rationale(gen_text):
    new_gen_text = [gt.split('[ANSWER]',1)[0].strip() for gt in gen_text]
    
    return new_gen_text

def lmap(f, x):
    """list(map(f, x))"""
    return list(map(f, x))