#!/usr/bin/env python3
"""Calculate AR, CR, NED, F1, BLEU and RO-ED from page predictions.

Input: JSONL {page_id, predict, label} with recognition line boundaries.
Output: one JSON file containing every metric. Legacy metric.json is accepted.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import numpy as np
from nltk.metrics import precision, recall, f_measure
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from Levenshtein import distance

def cal_ar_cr(task_id, data_list):

    def calculate_ar_cr(transcript, recognition_result):
        transcript_chars = list(transcript)
        result_chars = list(recognition_result)
        N = len(transcript_chars)
        if N == 0:
            return (0.0, 0.0)
        matrix = np.empty((len(transcript_chars) + 1, len(result_chars) + 1), dtype=np.int32)
        matrix[:, 0] = np.arange(len(transcript_chars) + 1)
        matrix[0, :] = np.arange(len(result_chars) + 1)
        result_codes = np.fromiter(map(ord, result_chars), dtype=np.int32)
        columns = np.arange(len(result_chars), dtype=np.int32)
        for i in range(1, len(transcript_chars) + 1):
            previous = matrix[i - 1]
            candidates = np.minimum(previous[1:] + 1, previous[:-1] + (result_codes != ord(transcript_chars[i - 1])))
            # Row recurrence: d[j] = min(candidate[j], d[j-1]+1).
            # Prefix minima compute the same integer distances in native code;
            # the historical substitution/deletion/insertion tie-break below stays exact.
            matrix[i, 1:] = np.minimum.accumulate(candidates - columns) + columns
        (i, j) = (len(transcript_chars), len(result_chars))
        substitutions = deletions = insertions = 0
        while i > 0 or j > 0:
            if i > 0 and j > 0 and (transcript_chars[i - 1] == result_chars[j - 1]):
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and (matrix[i][j] == matrix[i - 1][j - 1] + 1):
                substitutions += 1
                i -= 1
                j -= 1
            elif i > 0 and matrix[i][j] == matrix[i - 1][j] + 1:
                deletions += 1
                i -= 1
            elif j > 0 and matrix[i][j] == matrix[i][j - 1] + 1:
                insertions += 1
                j -= 1
        ar = (N - substitutions - deletions - insertions) / N
        cr = (N - substitutions - deletions) / N
        return (ar, cr)
    total_ar = 0.0
    total_cr = 0.0
    count = 0
    for item in data_list:
        transcript = item.get('Answer_Simplified', '')
        recognition_result = item.get('Response', '')
        (ar, cr) = calculate_ar_cr(transcript, recognition_result)
        total_ar += ar
        total_cr += cr
        count += 1
    average_ar = total_ar / count if count > 0 else 0.0
    average_cr = total_cr / count if count > 0 else 0.0
    print(f'任务ID：{task_id}, AR：{average_ar:.2%}, CR：{average_cr:.2%}')
    return (average_ar, average_cr)

def cal_edit_distance(task_id, data_list):
    if not data_list:
        return 0.0
    total_distance = 0.0
    for item in data_list:
        pred = item.get('Response', '')
        gt = item.get('Answer_Simplified', '')
        total_distance += distance(pred, gt) / max(len(pred), len(gt), 1)
    print(f'任务ID：{task_id}, Edit Distance：{total_distance / len(data_list):.2%}')
    return total_distance / len(data_list)

def cal_ocr_f1_p_r(task_id, data_list):
    f1_scores = []
    precision_scores = []
    recall_scores = []
    for item in data_list:
        answer = set(item['Answer_Simplified'])
        response = set(item['Response'])
        precision_score_value = precision(answer, response)
        recall_score_value = recall(answer, response) or 0.0
        f1_score_value = f_measure(answer, response)
        if precision_score_value is None:
            precision_score_value = 0.0
        if f1_score_value is None:
            f1_score_value = 0
        f1_scores.append(f1_score_value)
        precision_scores.append(precision_score_value)
        recall_scores.append(recall_score_value)
    avg_f1 = sum(f1_scores) / len(f1_scores)
    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)
    print(f'任务ID：{task_id}, F1-Score：{avg_f1:.2%}, Precision：{avg_precision:.2%}, Recall：{avg_recall:.2%}')
    return (avg_f1, avg_precision, avg_recall)

def cal_ocr_bleu(task_id, data_list):
    bleu_scores = []
    for item in data_list:
        reference = item['Answer_Simplified']
        hypothesis = item['Response']
        if isinstance(reference, list):
            reference = ''.join(reference)
        if isinstance(hypothesis, list):
            hypothesis = ''.join(hypothesis)
        reference_tokens = list(reference)
        hypothesis_tokens = list(hypothesis)
        smoothing_function = SmoothingFunction().method1
        bleu_score_value = sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smoothing_function)
        bleu_scores.append(bleu_score_value)
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f'任务ID：{task_id}, BLEU：{avg_bleu:.2%}')
    return avg_bleu


def read_records(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    rows = json.loads(text) if text.lstrip().startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows or not all(isinstance(r, dict) for r in rows):
        raise ValueError("evaluation input must contain nonempty object records")
    return rows


def cal_ro_ed(task_id, data_list, *, workers=12, output_dir=None, source_path=None):
    """Compute reading-order edit distance from records that preserve line boundaries."""
    if __package__:
        from .ro_ed.reading_order import calculate_reading_order
    else:
        from ro_ed.reading_order import calculate_reading_order
    records = [dict(page_id=str(r.get('page_id', r.get('Image_Name', r.get('ID', i)))),
                    label=r.get('label', r.get('Answer_Simplified', '')),
                    predict=r.get('predict', r.get('Response', '')))
               for i, r in enumerate(data_list)]
    summary = calculate_reading_order(records, workers, source_path=source_path, output_dir=output_dir)
    value = summary['ALL_page_avg']
    if value is None:
        raise ValueError('RO-ED has no nonempty order sequences to score')
    print(f'任务ID：{task_id}, RO-ED：{value:.2%}')
    return summary


def _character_records(rows):
    # Character metrics ignore line boundaries; RO-ED receives the original rows.
    def clean(text):
        text = re.sub(r"<(?:far_)?(?:left|right|up|down)(?:\|(?:far_)?(?:up|down))?>", "", text)
        text = text.replace(' ', '').replace('\n', '').replace('\r', '').replace('\u3000', '')
        return re.sub(r"<think>.*?</think>", '', text, flags=re.DOTALL)
    return [dict(Answer_Simplified=clean(r.get('label', r.get('Answer_Simplified', ''))),
                 Response=clean(r.get('predict', r.get('Response', '')))) for r in rows]


def main(input_path=None, *, ro_ed_input=None, workers=12, ro_ed_output=None,
         summary_json=None, ro_ed_only=False, skip_ro_ed=False):
    metrics = {}
    raw = read_records(input_path) if input_path is not None else None
    if not ro_ed_only:
        if input_path is None:
            raise ValueError('--input_path is required for character metrics')
        rows = _character_records(raw)
        ar, cr = cal_ar_cr(1, rows)
        ned = cal_edit_distance(1, rows)
        f1, p, r = cal_ocr_f1_p_r(1, rows)
        bleu = cal_ocr_bleu(1, rows)
        metrics.update({k: round(v * 100, 2) for k, v in
                        dict(AR=ar, CR=cr, NED=ned, F1=f1, Precision=p, Recall=r, BLEU=bleu).items()})
    if not skip_ro_ed:
        ro_rows = read_records(ro_ed_input) if ro_ed_input is not None else raw
        if ro_rows is None:
            raise ValueError('prediction input is required for RO-ED')
        ro = cal_ro_ed(1, ro_rows, workers=workers, output_dir=ro_ed_output,
                       source_path=ro_ed_input or input_path)
        metrics['RO-ED'] = ro['ALL_page_avg']
    if summary_json:
        path = Path(summary_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_path', required=True, help='page-prediction JSONL or legacy metric.json')
    parser.add_argument('--output_path', '--summary-json', dest='output_path', default='metrics.json',
                        help='combined JSON result (default: metrics.json)')
    parser.add_argument('--match-workers', type=int, default=12)
    parser.add_argument('--skip-ro-ed', action='store_true', help='calculate character metrics only')
    # Compatibility with existing local evaluation commands; unnecessary for new runs.
    parser.add_argument('--ro-ed-input', help=argparse.SUPPRESS)
    parser.add_argument('--ro-ed-output', help=argparse.SUPPRESS)
    parser.add_argument('--ro-ed-only', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.match_workers < 1:
        parser.error('--match-workers must be positive')
    main(args.input_path, ro_ed_input=args.ro_ed_input, workers=args.match_workers,
         ro_ed_output=args.ro_ed_output, summary_json=args.output_path,
         ro_ed_only=args.ro_ed_only, skip_ro_ed=args.skip_ro_ed)
