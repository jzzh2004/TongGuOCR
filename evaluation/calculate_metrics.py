import argparse
import contextlib
import json
import re
import nltk
import jieba
import logging
import torch
import numpy as np
import warnings
from collections import defaultdict
from zhconv import convert
from nltk.metrics import precision, recall, f_measure
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import Counter
from transformers import BertTokenizer, BertForMaskedLM
jieba.setLogLevel(logging.INFO)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

# Download link: https://huggingface.co/hfl/chinese-bert-wwm
model_path = "/path/to/chinese-bert-wwm"
# tokenizer = BertTokenizer.from_pretrained(model_path)
# model = BertForMaskedLM.from_pretrained(model_path)

id_to_task_dict_zh = {1:"文本OCR",2:"文白翻译",3:"句读",4:"命名实体识别",5:"字词解释",
                6:"阅读理解",7:"判断出处",8:"书法OCR",9:"书法鉴赏",10:"作者识别",
                11:"朝代识别",12:"标题识别",13:"字体识别",14:"流派识别",15:"作品介绍",
                16:"材质识别",17:"绘画鉴赏",18:"作者识别",19:"朝代识别",20:"标题识别",
                21:"画面问答",22:"背景介绍",23:"绘画技法",24:"出处识别",25:"绘画描述",
                26:"绘画OCR",27:"作品介绍",28:"甲骨文OCR",29:"象形解读",30:"印鉴OCR",
                31:"人物识别",32:"名称识别",33:"朝代识别",34:"馆藏识别",35:"文物介绍",
                36:"文物分类",37:"插图OCR",38:"插图描述",39:"实体介绍",40:"图像诗句匹配",
                41:"出处识别",42:"主题分类",43:"图人匹配",44:"情节介绍",45:"画面问答"
                }

id_to_domain_dict_zh = {1:"古籍文本",2:"古籍文本",3:"古籍文本",4:"古籍文本",5:"古籍文本",
                6:"古籍文本",7:"古籍文本",8:"书法",9:"书法",10:"书法",
                11:"书法",12:"书法",13:"书法",14:"书法",15:"书法",
                16:"绘画",17:"绘画",18:"绘画",19:"绘画",20:"绘画",
                21:"绘画",22:"绘画",23:"绘画",24:"绘画",25:"绘画",
                26:"绘画",27:"绘画",28:"甲骨文",29:"甲骨文",30:"印鉴",
                31:"印鉴",32:"文物",33:"文物",34:"文物",35:"文物",
                36:"文物",37:"插图",38:"插图",39:"插图",40:"插图",
                41:"插图",42:"插图",43:"插图",44:"插图",45:"插图"
                }

def clean_text(text):
    cleaned_text = re.sub(r'[^\u4e00-\u9fff\w]', '', text)
    return cleaned_text
    
def cal_acc(task_id, data_list):
    total_correct = 0
    total_questions = len(data_list)
    for item in data_list:
        response = clean_text(item["Response"])
        response = convert(response, locale='zh-cn')
        answer = item["Answer_Simplified"]
        if answer in response:
            total_correct += 1
    average_accuracy = (total_correct / total_questions if total_questions > 0 else 0)
    # print(
    #     f"任务ID：{task_id}, 任务名称：{id_to_task_dict_zh[task_id]}, 领域：{id_to_domain_dict_zh[task_id]}, Acc：{average_accuracy:.2%}"
    # )
    print(
        f"任务ID：{task_id}, Acc：{average_accuracy:.2%}"
    )

    return average_accuracy

def cal_ar_cr(task_id, data_list):
    def calculate_ar_cr(transcript, recognition_result):
        transcript_chars = list(transcript)
        result_chars = list(recognition_result)
        N = len(transcript_chars)
        if N == 0:
            return 0.0, 0.0
        matrix = [[0] * (len(result_chars) + 1) for _ in range(len(transcript_chars) + 1)]
        for i in range(len(transcript_chars) + 1):
            matrix[i][0] = i
        for j in range(len(result_chars) + 1):
            matrix[0][j] = j
        for i in range(1, len(transcript_chars) + 1):
            for j in range(1, len(result_chars) + 1):
                if transcript_chars[i-1] == result_chars[j-1]:
                    matrix[i][j] = matrix[i-1][j-1]
                else:
                    matrix[i][j] = min(
                        matrix[i-1][j] + 1,    # deletion
                        matrix[i][j-1] + 1,    # insertion
                        matrix[i-1][j-1] + 1   # substitution
                    )
        i, j = len(transcript_chars), len(result_chars)
        substitutions = deletions = insertions = 0
        while i > 0 or j > 0:
            if i > 0 and j > 0 and transcript_chars[i-1] == result_chars[j-1]:
                i -= 1
                j -= 1
            else:
                if i > 0 and j > 0 and matrix[i][j] == matrix[i-1][j-1] + 1:
                    substitutions += 1
                    i -= 1
                    j -= 1
                elif i > 0 and matrix[i][j] == matrix[i-1][j] + 1:
                    deletions += 1
                    i -= 1
                elif j > 0 and matrix[i][j] == matrix[i][j-1] + 1:
                    insertions += 1
                    j -= 1
        ar = (N - substitutions - deletions - insertions) / N
        cr = (N - substitutions - deletions) / N
        return ar, cr
    
    total_ar = 0.0
    total_cr = 0.0
    count = 0

    for item in data_list:
        transcript = item.get("Answer_Simplified", "")
        # transcript = clean_text(transcript)
        # transcript = convert(transcript, locale='zh-cn')
        recognition_result = item.get("Response", "")
        # recognition_result = clean_text(recognition_result)
        # recognition_result = convert(recognition_result, locale='zh-cn')
        ar, cr = calculate_ar_cr(transcript, recognition_result)
        total_ar += ar
        total_cr += cr
        count += 1
    average_ar = total_ar / count if count > 0 else 0.0
    average_cr = total_cr / count if count > 0 else 0.0
    # print(
    #     f"任务ID：{task_id}, 任务名称：{id_to_task_dict_zh[task_id]}, 领域：{id_to_domain_dict_zh[task_id]}, AR：{average_ar:.2%}, CR：{average_cr:.2%}"
    # )
    print(
        f"任务ID：{task_id}, AR：{average_ar:.2%}, CR：{average_cr:.2%}"
    )
    return average_ar, average_cr

def cal_edit_distance(task_id, data_list):
    if not data_list:
        return 0.0
    total_distance = 0.0
    for item in data_list:
        pred = item.get("Response", "")
        # pred = clean_text(pred)
        # pred = convert(pred, locale='zh-cn')
        gt = item.get("Answer_Simplified", "")
        # gt = clean_text(gt)
        # gt = convert(gt, locale='zh-cn')
        total_distance += nltk.edit_distance(pred, gt) / max(len(pred), len(gt))
    # print(
    #     f"任务ID：{task_id}, 任务名称：{id_to_task_dict_zh[task_id]}, 领域：{id_to_domain_dict_zh[task_id]}, Edit Distance：{total_distance / len(data_list):.2%}"
    # )
    print(
        f"任务ID：{task_id}, Edit Distance：{total_distance / len(data_list):.2%}"
    )
    return total_distance / len(data_list)

def cal_ocr_f1_p_r(task_id, data_list):
    f1_scores = []
    precision_scores = []
    recall_scores = []
    
    for item in data_list:


        answer = set((item['Answer_Simplified']))
        response = set((item['Response']))
        precision_score_value = precision(answer, response)
        recall_score_value = recall(answer, response)
        f1_score_value = f_measure(answer, response)

        if precision_score_value is None:
            precision_score_value = 0.0
        if f1_score_value is None:
            f1_score_value = 0
        
        # print(item["ID"],precision_score_value,recall_score_value,f1_score_value)
        f1_scores.append(f1_score_value)
        precision_scores.append(precision_score_value)
        recall_scores.append(recall_score_value)
    
    avg_f1 = sum(f1_scores) / len(f1_scores)
    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)
    # print(
    #     f"任务ID：{task_id}, 任务名称：{id_to_task_dict_zh[task_id]}, 领域：{id_to_domain_dict_zh[task_id]}, F1-Score：{avg_f1:.2%}, Precision：{avg_precision:.2%}, Recall：{avg_recall:.2%}"
    # )
    print(
        f"任务ID：{task_id}, F1-Score：{avg_f1:.2%}, Precision：{avg_precision:.2%}, Recall：{avg_recall:.2%}"
    )
    return avg_f1, avg_precision, avg_recall

def cal_ocr_bleu(task_id, data_list):
    bleu_scores = []
    for item in data_list:
        reference = item['Answer_Simplified']
        # reference = clean_text(reference)
        # reference = convert(reference, locale='zh-cn')
        hypothesis = item['Response']
        # hypothesis = clean_text(hypothesis)
        # hypothesis = convert(hypothesis, locale='zh-cn')
        
        if isinstance(reference, list):
            reference = "".join(reference)
        if isinstance(hypothesis, list):
            hypothesis = "".join(hypothesis)
        
        reference_tokens = list(reference)
        hypothesis_tokens = list(hypothesis)
        
        smoothing_function = SmoothingFunction().method1
        bleu_score_value = sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smoothing_function)
        
        bleu_scores.append(bleu_score_value)

    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    # print(
    #     f"任务ID：{task_id}, 任务名称：{id_to_task_dict_zh[task_id]}, 领域：{id_to_domain_dict_zh[task_id]}, BLEU：{avg_bleu:.2%}"
    # )
    print(
        f"任务ID：{task_id}, BLEU：{avg_bleu:.2%}"
    )
    return avg_bleu

def cal_bleu(task_id, data_list):
    smoothing_function = SmoothingFunction().method4
    bleu_scores = []
    
    for data in data_list:
        reference = [data["Answer_Simplified"]]
        hypothesis = data["Response"]
        # hypothesis = convert(hypothesis, locale='zh-cn')
        # reference = convert(reference[0], locale='zh-cn')
    
        reference_tokens = list(jieba.cut(reference[0]))
        hypothesis_tokens = list(jieba.cut(hypothesis))
        
        bleu_score_value = sentence_bleu([reference_tokens], hypothesis_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)
        bleu_scores.append(bleu_score_value)
    
    average_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    # print(
    #     f"任务ID：{task_id}, 任务名称：{id_to_task_dict_zh[task_id]}, 领域：{id_to_domain_dict_zh[task_id]}, BLEU：{average_bleu:.2%}"
    # )
    print(
        f"任务ID：{task_id}, BLEU：{average_bleu:.2%}"
    )
    return average_bleu

def lcs(x, y):
    m = len(x)
    n = len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

def cal_rouge(task_id, data_list):
    f1_1_scores = []
    f1_2_scores = []
    f1_l_scores = []
    
    for item in data_list:
        reference = item['Answer_Simplified'].replace(" ", "")
        hypothesis = item['Response'].replace(" ", "")
        hypothesis = convert(hypothesis, locale="zh-cn")
        
        # ROUGE-1
        reference_1 = Counter(reference)
        hypothesis_1 = Counter(hypothesis)
        overlap_1 = sum((reference_1 & hypothesis_1).values())
        precision_1 = overlap_1 / sum(hypothesis_1.values()) if sum(hypothesis_1.values()) > 0 else 0
        recall_1 = overlap_1 / sum(reference_1.values()) if sum(reference_1.values()) > 0 else 0
        f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1) if precision_1 + recall_1 > 0 else 0

        # ROUGE-2
        reference_2 = [reference[i:i+2] for i in range(len(reference) - 1)]
        hypothesis_2 = [hypothesis[i:i+2] for i in range(len(hypothesis) - 1)]
        reference_2_count = Counter(reference_2)
        hypothesis_2_count = Counter(hypothesis_2)
        overlap_2 = sum((reference_2_count & hypothesis_2_count).values())
        precision_2 = overlap_2 / sum(hypothesis_2_count.values()) if sum(hypothesis_2_count.values()) > 0 else 0
        recall_2 = overlap_2 / sum(reference_2_count.values()) if sum(reference_2_count.values()) > 0 else 0
        f1_2 = 2 * precision_2 * recall_2 / (precision_2 + recall_2) if precision_2 + recall_2 > 0 else 0

        # ROUGE-L
        lcs_length = lcs(reference, hypothesis)
        precision_l = lcs_length / len(hypothesis) if len(hypothesis) > 0 else 0
        recall_l = lcs_length / len(reference) if len(reference) > 0 else 0
        f1_l = 2 * precision_l * recall_l / (precision_l + recall_l) if precision_l + recall_l > 0 else 0

        f1_1_scores.append(f1_1)
        f1_2_scores.append(f1_2)
        f1_l_scores.append(f1_l)

    avg_f1_1 = sum(f1_1_scores) / len(f1_1_scores) if f1_1_scores else 0
    avg_f1_2 = sum(f1_2_scores) / len(f1_2_scores) if f1_2_scores else 0
    avg_f1_l = sum(f1_l_scores) / len(f1_l_scores) if f1_l_scores else 0

    # print(
    #     f"任务ID：{task_id}, 任务名称：{id_to_task_dict_zh[task_id]}, 领域：{id_to_domain_dict_zh[task_id]}, ROUGE-1：{avg_f1_1:.2%}, ROUGE-2：{avg_f1_2:.2%}, ROUGE-L：{avg_f1_l:.2%}"
    # )
    print(
        f"任务ID：{task_id}, ROUGE-1：{avg_f1_1:.2%}, ROUGE-2：{avg_f1_2:.2%}, ROUGE-L：{avg_f1_l:.2%}"
    )
    return avg_f1_1, avg_f1_2, avg_f1_l

def cal_punctuation_p_r_f1(task_id, data_list):
    def calculate_punctuation_metrics(prediction, reference):
        rule = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fa5]")
        result_puncs = [prediction[i-1] + prediction[i] for i in [m.span()[0] for m in re.finditer(rule, prediction)]]
        label_puncs = [reference[i-1] + reference[i] for i in [m.span()[0] for m in re.finditer(rule, reference)]]
        TP = 0
        for pred in result_puncs:
            if pred in label_puncs:
                TP += 1
        precision = TP / len(result_puncs) if len(result_puncs) > 0 else 0
        recall = TP / len(label_puncs) if len(label_puncs) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        return precision, recall, f1
    
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for item in data_list:
        prediction = item['Response']
        prediction = convert(prediction, locale='zh-cn')
        reference = item['Answer_Simplified']
        
        precision, recall, f1 = calculate_punctuation_metrics(prediction, reference)
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    # print(
    #     f"任务ID：{task_id}, 任务名称：{id_to_task_dict_zh[task_id]}, 领域：{id_to_domain_dict_zh[task_id]}, Precision：{avg_precision:.2%}, Recall：{avg_recall:.2%}, F1-Score：{avg_f1:.2%}"
    # )
    print(
        f"任务ID：{task_id}, Precision：{avg_precision:.2%}, Recall：{avg_recall:.2%}, F1-Score：{avg_f1:.2%}"
    )
    return avg_precision, avg_recall, avg_f1

def cal_ner_p_r_f1(task_id, data_list):
    def calculate_ner_metrics(prediction, reference):
        result_entities = set(prediction.strip().split('、'))
        label_entities = set(reference.split('、'))
        TP = len(result_entities.intersection(label_entities))
        precision = TP / len(result_entities) if len(result_entities) > 0 else 0
        recall = TP / len(label_entities) if len(label_entities) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        return precision, recall, f1

    precision_scores = []
    recall_scores = []
    f1_scores = []

    for item in data_list:
        prediction = item['Response']
        prediction = convert(prediction, locale='zh-cn')
        reference = item['Answer_Simplified']
        
        precision, recall, f1 = calculate_ner_metrics(prediction, reference)
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    # print(
    #     f"任务ID：{task_id}, 任务名称：{id_to_task_dict_zh[task_id]}, 领域：{id_to_domain_dict_zh[task_id]}, Precision：{avg_precision:.2%}, Recall：{avg_recall:.2%}, F1-Score：{avg_f1:.2%}"
    # )
    print(
        f"任务ID：{task_id}, Precision：{avg_precision:.2%}, Recall：{avg_recall:.2%}, F1-Score：{avg_f1:.2%}"
    )
    return avg_precision, avg_recall, avg_f1

def split_text(text, max_length=512):
    tokens = tokenizer.encode(text, truncation=False)
    return [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]

def get_bert_embeddings(text):
    text_blocks = split_text(text)
    embeddings = []
    for block in text_blocks:
        input_ids = torch.tensor([block])
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            last_hidden_state = hidden_states[-1]
            cls_embedding = last_hidden_state[:, 0, :]
            embeddings.append(cls_embedding.squeeze().cpu().numpy())
    return np.mean(embeddings, axis=0)

def cal_bertscore(task_id, data_list):
    bertscores = []
    
    for item in data_list:
        ref_text = item['Answer_Simplified']
        hyp_text = item['Response']
        hyp_text = convert(hyp_text, locale='zh-cn')
        
        ref_embedding = get_bert_embeddings(ref_text)
        hyp_embedding = get_bert_embeddings(hyp_text)
        
        cos_sim = np.dot(ref_embedding, hyp_embedding) / (np.linalg.norm(ref_embedding) * np.linalg.norm(hyp_embedding))
        bertscores.append(cos_sim)
    
    avg_bertscore = np.mean(bertscores)
    # print(
    #     f"任务ID：{task_id}, 任务名称：{id_to_task_dict_zh[task_id]}, 领域：{id_to_domain_dict_zh[task_id]}, BERTScore：{avg_bertscore:.2%}"
    # )
    print(
        f"任务ID：{task_id}, BERTScore：{avg_bertscore:.2%}"
    )
    return avg_bertscore

def cal_anls(task_id, data_list):
    def levenshtein_distance(s1, s2):
        len_s1, len_s2 = len(s1), len(s2)
        dp = np.zeros((len_s1 + 1, len_s2 + 1), dtype=int)

        for i in range(len_s1 + 1):
            dp[i][0] = i
        for j in range(len_s2 + 1):
            dp[0][j] = j

        for i in range(1, len_s1 + 1):
            for j in range(1, len_s2 + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1,    
                               dp[i][j - 1] + 1,  
                               dp[i - 1][j - 1] + cost) 
        
        return dp[len_s1][len_s2]

    def anls(s1, s2):
        lev_dist = levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        if max_len == 0:  
            return 1.0
        return 1 - lev_dist / max_len

    anls_scores = []
    for item in data_list:
        ref_text = item['Answer_Simplified']
        hyp_text = item['Response']
        hyp_text = convert(hyp_text, locale='zh-cn')

        score = anls(ref_text, hyp_text)
        anls_scores.append(score)
    
    avg_anls = np.mean(anls_scores)
    # print(
    #     f"任务ID：{task_id}, 任务名称：{id_to_task_dict_zh[task_id]}, 领域：{id_to_domain_dict_zh[task_id]}, ANLS：{avg_anls:.2%}"
    # )
    print(
        f"任务ID：{task_id}, ANLS：{avg_anls:.2%}"
    )
    return avg_anls

def main(input_path):
    print(input_path)
    with open(input_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    task_groups = defaultdict(list)
    for item in data_list:
        task_groups[item["Task_ID"]].append(item)

    print("----------------------------------------------------------------")
    cal_ar_cr(1,task_groups[1])
    cal_edit_distance(1,task_groups[1])
    cal_ocr_f1_p_r(1,task_groups[1])
    cal_ocr_bleu(1,task_groups[1])

# cal_bleu(2,task_groups[2])
# cal_rouge(2,task_groups[2])

# cal_punctuation_p_r_f1(3,task_groups[3])

# cal_ner_p_r_f1(4,task_groups[4])

# cal_acc(5,task_groups[5])

# cal_bertscore(6,task_groups[6])
# cal_anls(6,task_groups[6])

# cal_acc(7,task_groups[7])

# print("----------------------------------------------------------------")

# cal_ar_cr(8,task_groups[8])
# cal_edit_distance(8,task_groups[8])
# cal_ocr_f1_p_r(8,task_groups[8])
# cal_ocr_bleu(8,task_groups[8])

# cal_bertscore(9,task_groups[9])
# cal_anls(9,task_groups[9])

# cal_acc(10,task_groups[10])

# cal_acc(11,task_groups[11])

# cal_acc(12,task_groups[12])

# cal_acc(13,task_groups[13])

# cal_acc(14,task_groups[14])

# cal_bertscore(15,task_groups[15])
# cal_anls(15,task_groups[15])

# print("----------------------------------------------------------------")

# cal_acc(16,task_groups[16])

# cal_bertscore(17,task_groups[17])
# cal_anls(17,task_groups[17])

# cal_acc(18,task_groups[18])

# cal_acc(19,task_groups[19])

# cal_acc(20,task_groups[20])

# cal_acc(21,task_groups[21])

# cal_bertscore(22,task_groups[22])
# cal_anls(22,task_groups[22])

# cal_acc(23,task_groups[23])

# cal_acc(24,task_groups[24])

# cal_bertscore(25,task_groups[25])
# cal_anls(25,task_groups[25])

# cal_ar_cr(26,task_groups[26])
# cal_edit_distance(26,task_groups[26])
# cal_ocr_f1_p_r(26,task_groups[26])
# cal_ocr_bleu(26,task_groups[26])

# cal_bertscore(27,task_groups[27])
# cal_anls(27,task_groups[27])

# print("----------------------------------------------------------------")

# cal_ar_cr(28,task_groups[28])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as f:
            with contextlib.redirect_stdout(f):
                main(args.input_path)
    else:
        main(args.input_path)
# cal_edit_distance(28,task_groups[28])
# cal_ocr_f1_p_r(28,task_groups[28])
# cal_ocr_bleu(28,task_groups[28])

# cal_bertscore(29,task_groups[29])
# cal_anls(29,task_groups[29])

# print("----------------------------------------------------------------")

# cal_ar_cr(30,task_groups[30])
# cal_edit_distance(30,task_groups[30])
# cal_ocr_f1_p_r(30,task_groups[30])
# cal_ocr_bleu(30,task_groups[30])

# cal_acc(31,task_groups[31])

# print("----------------------------------------------------------------")

# cal_acc(32,task_groups[32])

# cal_acc(33,task_groups[33])

# cal_acc(34,task_groups[34])

# cal_bertscore(35,task_groups[35])
# cal_anls(35,task_groups[35])

# cal_acc(36,task_groups[36])

# print("----------------------------------------------------------------")

# cal_ar_cr(37,task_groups[37])
# cal_edit_distance(37,task_groups[37])
# cal_ocr_f1_p_r(37,task_groups[37])
# cal_ocr_bleu(37,task_groups[37])

# cal_bertscore(38,task_groups[38])
# cal_anls(38,task_groups[38])

# cal_bertscore(39,task_groups[39])
# cal_anls(39,task_groups[39])

# cal_acc(40,task_groups[40])

# cal_acc(41,task_groups[41])

# cal_acc(42,task_groups[42])

# cal_acc(43,task_groups[43])

# cal_bertscore(44,task_groups[44])
# cal_anls(44,task_groups[44])

# cal_acc(45,task_groups[45])
