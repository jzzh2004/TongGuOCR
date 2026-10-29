# Copyright (c) OpenDataLab. Adapted for TongGuOCR under Apache-2.0.
# Source: OmniDocBench be7a44d371e43801fabd2e31e02976f106b6aa16.
# Text-only matching path; see THIRD_PARTY_NOTICES.md.
"""Standalone text-line matching for TongGuOCR's reading-order metric.

Hungarian assignment, truncated-line merging, fuzzy matching, and timeout
fallback retain the archived protocol's decisions. This module accepts only
text_block GT and text_all predictions, constructed by reading_order.py.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import copy


import time
import logging
from typing import Any, Dict, List
import numpy as np
import Levenshtein
from Levenshtein import distance as Levenshtein_distance
from scipy.optimize import linear_sum_assignment
from .normalize import normalized_text, normalized_formula, strip_formula_delimiters
logger = logging.getLogger(__name__)

def get_gt_pred_lines(gt_items, pred_items, line_type=None):
    gt_lines = [str(item['text']) for item in gt_items]
    pred_lines = [str(item['content']) for item in pred_items]
    gt_filtered = [(a, normalized_text(a), item['category_type']) for (a, item) in zip(gt_lines, gt_items)]
    gt_filtered = [(a, b, c) for (a, b, c) in gt_filtered if a and b]
    pred_filtered = [(a, normalized_text(a)) for a in pred_lines]
    pred_filtered = [(a, b) for (a, b) in pred_filtered if a and b]
    return ([a for (a, b, c) in gt_filtered], [b for (a, b, c) in gt_filtered], [c for (a, b, c) in gt_filtered], [a for (a, b) in pred_filtered], [b for (a, b) in pred_filtered], gt_items, pred_items)

class MatchResult(list):

    def __init__(self, iterable=(), fallback_reason=None):
        super().__init__(iterable)
        self.fallback_reason = fallback_reason

def get_pred_category_type(pred_idx, pred_items):
    if pred_items[pred_idx].get('fine_category_type'):
        pred_pred_category_type = pred_items[pred_idx]['fine_category_type']
    else:
        pred_pred_category_type = pred_items[pred_idx]['category_type']
    return pred_pred_category_type

MAX_TRUNCATED_PRED_MERGE = 160

class TruncatedMatchTimeout(RuntimeError):
    pass

def compute_edit_distance_matrix_new(gt_lines, matched_lines):
    try:
        distance_matrix = np.zeros((len(gt_lines), len(matched_lines)))
        for (i, gt_line) in enumerate(gt_lines):
            for (j, matched_line) in enumerate(matched_lines):
                if len(gt_line) == 0 and len(matched_line) == 0:
                    distance_matrix[i][j] = 0
                else:
                    distance_matrix[i][j] = Levenshtein.distance(gt_line, matched_line) / max(len(matched_line), len(gt_line))
        return distance_matrix
    except ZeroDivisionError:
        raise

TEXT_GT_CATEGORIES = {'text_block', 'title', 'code_txt', 'code_txt_caption', 'reference', 'equation_caption', 'figure_caption', 'figure_footnote', 'table_caption', 'table_footnote', 'code_algorithm', 'code_algorithm_caption', 'header', 'footer', 'page_footnote', 'page_number'}

def _position_sort_key(item):
    order = item.get('order')
    if isinstance(order, (int, float)):
        return (0, order)
    position = item.get('position', [None])
    if isinstance(position, list) and position:
        first = position[0]
        if isinstance(first, (int, float)):
            return (1, first)
        if isinstance(first, (list, tuple)) and first and isinstance(first[0], (int, float)):
            return (1, first[0])
    return (2, 0)

def _timeout_match_threshold(norm_text):
    length = len(norm_text or '')
    if length <= 6:
        return 0.4
    if length <= 20:
        return 0.55
    if length <= 80:
        return 0.68
    return 0.78

def _build_timeout_fallback_gt_records(gt_items):
    records = []
    for (source_idx, item) in enumerate(gt_items or []):
        gt_category = item.get('fine_category_type') or item.get('category_type') or ''
        if item.get('content'):
            raw_text = str(item['content'])
            norm_text = normalized_text(raw_text) if item.get('category_type') == 'text_all' else raw_text
        elif item.get('category_type') in TEXT_GT_CATEGORIES:
            raw_text = str(item.get('text', ''))
            norm_text = normalized_text(raw_text)
        elif item.get('category_type') == 'equation_isolated':
            raw_text = str(item.get('latex', ''))
            norm_text = normalized_formula(raw_text)
        else:
            continue
        if not raw_text or not norm_text:
            continue
        records.append({'item': item, 'source_idx': source_idx, 'raw': raw_text, 'norm': norm_text, 'category': gt_category, 'position': [item.get('order') if item.get('order') else item.get('position', [''])[0]], 'attribute': [item.get('attribute', {})]})
    return records

def _build_timeout_fallback_pred_records(pred_items):
    records = []
    for (source_idx, item) in enumerate(pred_items or []):
        pred_category = item.get('fine_category_type') or item.get('category_type') or ''
        raw_text = str(item.get('content', ''))
        if not raw_text:
            continue
        if item.get('category_type') == 'text_all':
            norm_text = normalized_text(raw_text)
        elif item.get('category_type') == 'equation_isolated':
            norm_text = normalized_formula(raw_text)
        else:
            norm_text = raw_text
        if not norm_text:
            continue
        records.append({'item': item, 'source_idx': source_idx, 'raw': raw_text, 'norm': norm_text, 'category': pred_category, 'position': item.get('position', [''])[0] if item.get('position') else ''})
    return sorted(records, key=lambda record: _position_sort_key(record['item']))

def _safe_positive_int(value, default, minimum=1, maximum=None):
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value

def _safe_non_negative_float(value, default):
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    return max(0.0, value)

def _record_is_formula(record):
    item = record.get('item', {})
    if item.get('category_type') == 'equation_isolated':
        return True
    return record.get('category') in {'equation_isolated', 'equation_inline', 'formula_rescue'}

def _record_is_text_like(record):
    return not _record_is_formula(record)

def _is_open_ended_text(raw_text):
    text = str(raw_text or '').rstrip()
    if not text:
        return False
    return text[-1] not in '.!?。！？;；:：)]}」』”’"'

def _starts_with_continuation(raw_text):
    text = str(raw_text or '').lstrip()
    if not text:
        return False
    return text[0].islower() or text[0].isdigit() or text[0] in ',.;:!?)]}%'

def _resolve_chunked_hungarian_params(gt_records, short_line_max_chars=None, target_chunk_chars=None, max_chunk_chars=None, max_chunk_span=8, order_window=None, order_penalty=0.08):
    gt_text_lengths = [len(record['norm']) for record in gt_records if record.get('norm') and _record_is_text_like(record)]
    if gt_text_lengths:
        gt_median_len = int(np.median(gt_text_lengths))
    else:
        gt_median_len = 120
    short_line_max_chars = _safe_positive_int(short_line_max_chars, default=max(24, min(72, int(gt_median_len * 0.45))), minimum=12, maximum=160)
    target_chunk_chars = _safe_positive_int(target_chunk_chars, default=max(short_line_max_chars * 2, min(220, int(gt_median_len * 0.95))), minimum=short_line_max_chars, maximum=420)
    max_chunk_chars = _safe_positive_int(max_chunk_chars, default=max(target_chunk_chars + 48, min(320, int(gt_median_len * 1.35))), minimum=target_chunk_chars, maximum=640)
    max_chunk_span = _safe_positive_int(max_chunk_span, default=8, minimum=1, maximum=32)
    order_penalty = _safe_non_negative_float(order_penalty, default=0.08)
    if order_window is None:
        order_window = None
    else:
        order_window = _safe_positive_int(order_window, default=12, minimum=2, maximum=128)
    return {'short_line_max_chars': short_line_max_chars, 'target_chunk_chars': target_chunk_chars, 'max_chunk_chars': max_chunk_chars, 'max_chunk_span': max_chunk_span, 'order_window': order_window, 'order_penalty': order_penalty}

def _make_pred_chunk(span_records):
    raw_parts = [record['raw'] for record in span_records if record.get('raw')]
    norm_parts = [record['norm'] for record in span_records if record.get('norm')]
    if not raw_parts or not norm_parts:
        return None
    if all((record.get('category') == span_records[0].get('category') for record in span_records)):
        category = span_records[0].get('category', '')
    elif all((_record_is_text_like(record) for record in span_records)):
        category = 'text_block'
    else:
        category = span_records[0].get('category', '')
    return {'pred_indices': [record['source_idx'] for record in span_records], 'raw': ' '.join(raw_parts), 'norm': ''.join(norm_parts), 'category': category, 'position': span_records[0].get('position', ''), 'records': span_records}

def _build_timeout_fallback_pred_chunks(pred_records, chunk_params):
    if not pred_records:
        return []
    short_line_max_chars = chunk_params['short_line_max_chars']
    target_chunk_chars = chunk_params['target_chunk_chars']
    max_chunk_chars = chunk_params['max_chunk_chars']
    max_chunk_span = chunk_params['max_chunk_span']
    pred_chunks = []
    idx = 0
    while idx < len(pred_records):
        current_record = pred_records[idx]
        span_records = [current_record]
        idx += 1
        if _record_is_text_like(current_record):
            current_norm_len = len(current_record.get('norm', ''))
            while idx < len(pred_records):
                next_record = pred_records[idx]
                if not _record_is_text_like(next_record):
                    break
                if len(span_records) >= max_chunk_span:
                    break
                next_norm_len = len(next_record.get('norm', ''))
                merged_norm_len = current_norm_len + next_norm_len
                if merged_norm_len > max_chunk_chars:
                    break
                should_merge = current_norm_len < target_chunk_chars or next_norm_len <= short_line_max_chars or _is_open_ended_text(span_records[-1].get('raw', '')) or _starts_with_continuation(next_record.get('raw', ''))
                if not should_merge:
                    break
                span_records.append(next_record)
                idx += 1
                current_norm_len = merged_norm_len
                if current_norm_len >= target_chunk_chars and (not _is_open_ended_text(span_records[-1].get('raw', ''))):
                    break
        chunk = _make_pred_chunk(span_records)
        if chunk is not None:
            pred_chunks.append(chunk)
    return pred_chunks

def _build_chunked_cost_matrices(gt_records, pred_chunks, order_window=None, order_penalty=0.08):
    gt_norms = [record['norm'] for record in gt_records]
    pred_norms = [chunk['norm'] for chunk in pred_chunks]
    edit_matrix = compute_edit_distance_matrix_new(gt_norms, pred_norms)
    cost_matrix = np.array(edit_matrix, copy=True)
    gt_count = len(gt_records)
    pred_count = len(pred_chunks)
    if gt_count <= 1 or pred_count <= 1:
        normalized_gt_den = 1
        normalized_pred_den = 1
    else:
        normalized_gt_den = gt_count - 1
        normalized_pred_den = pred_count - 1
    for (gt_idx, gt_record) in enumerate(gt_records):
        gt_is_formula = _record_is_formula(gt_record)
        expected_pred_idx = 0
        if pred_count > 1 and gt_count > 1:
            expected_pred_idx = round(gt_idx * normalized_pred_den / normalized_gt_den)
        for (pred_idx, pred_chunk) in enumerate(pred_chunks):
            penalty = 0.0
            pred_is_formula = any((_record_is_formula(record) for record in pred_chunk['records']))
            if gt_is_formula and (not pred_is_formula):
                penalty += 0.18
            elif not gt_is_formula and pred_is_formula:
                penalty += 0.1
            if order_window is not None:
                overflow = abs(pred_idx - expected_pred_idx) - order_window
                if overflow > 0:
                    penalty += 0.3 + 0.02 * overflow
            else:
                normalized_gap = abs(gt_idx / normalized_gt_den - pred_idx / normalized_pred_den)
                penalty += order_penalty * normalized_gap
            cost_matrix[gt_idx][pred_idx] += penalty
    return (edit_matrix, cost_matrix)

def match_gt2pred_timeout_safe(gt_items, pred_items, img_name, short_line_max_chars=None, target_chunk_chars=None, max_chunk_chars=None, max_chunk_span=8, order_window=None, order_penalty=0.08, fallback_reason=None):
    gt_records = _build_timeout_fallback_gt_records(gt_items)
    pred_records = _build_timeout_fallback_pred_records(pred_items)
    match_list = []
    chunk_params = _resolve_chunked_hungarian_params(gt_records, short_line_max_chars=short_line_max_chars, target_chunk_chars=target_chunk_chars, max_chunk_chars=max_chunk_chars, max_chunk_span=max_chunk_span, order_window=order_window, order_penalty=order_penalty)
    pred_chunks = _build_timeout_fallback_pred_chunks(pred_records, chunk_params)
    if chunk_params['order_window'] is None and pred_chunks:
        chunk_params['order_window'] = max(6, min(48, len(pred_chunks) // 6 + 6))
    print(f"[timeout-fallback] {img_name}: gt={len(gt_records)} pred={len(pred_records)} chunk={len(pred_chunks)} short<={chunk_params['short_line_max_chars']} target={chunk_params['target_chunk_chars']} max={chunk_params['max_chunk_chars']} span={chunk_params['max_chunk_span']} window={chunk_params['order_window']}", flush=True)

    def append_unmatched_gt(gt_record):
        match_list.append({'gt_idx': [gt_record['source_idx']], 'gt': gt_record['raw'], 'norm_gt': gt_record['norm'], 'gt_category_type': gt_record['category'], 'gt_position': gt_record['position'], 'gt_attribute': gt_record['attribute'], 'pred_idx': [''], 'pred': '', 'norm_pred': '', 'pred_category_type': '', 'pred_position': '', 'edit': 1, 'img_id': img_name})

    def append_unmatched_pred(pred_chunk):
        match_list.append({'gt_idx': [''], 'gt': '', 'norm_gt': '', 'gt_category_type': '', 'gt_position': [''], 'gt_attribute': [{}], 'pred_idx': pred_chunk['pred_indices'], 'pred': pred_chunk['raw'], 'norm_pred': pred_chunk['norm'], 'pred_category_type': pred_chunk['category'], 'pred_position': pred_chunk['position'], 'edit': 1, 'img_id': img_name})
    if not gt_records:
        for pred_chunk in pred_chunks:
            append_unmatched_pred(pred_chunk)
        return MatchResult(match_list, fallback_reason=fallback_reason)
    if not pred_chunks:
        for gt_record in gt_records:
            append_unmatched_gt(gt_record)
        return MatchResult(match_list, fallback_reason=fallback_reason)
    (edit_matrix, cost_matrix) = _build_chunked_cost_matrices(gt_records, pred_chunks, order_window=chunk_params['order_window'], order_penalty=chunk_params['order_penalty'])
    (row_ind, col_ind) = linear_sum_assignment(cost_matrix)
    row_to_col = {int(row): int(col) for (row, col) in zip(row_ind, col_ind)}
    matched_chunk_indices = set()
    for (gt_idx, gt_record) in enumerate(gt_records):
        pred_chunk_idx = row_to_col.get(gt_idx)
        if pred_chunk_idx is None:
            append_unmatched_gt(gt_record)
            continue
        pred_chunk = pred_chunks[pred_chunk_idx]
        pure_edit = float(edit_matrix[gt_idx][pred_chunk_idx])
        threshold = _timeout_match_threshold(gt_record['norm'])
        gt_is_formula = _record_is_formula(gt_record)
        pred_is_formula = any((_record_is_formula(record) for record in pred_chunk['records']))
        if gt_is_formula != pred_is_formula:
            threshold -= 0.08
        threshold = max(0.25, threshold)
        if pure_edit <= threshold:
            matched_chunk_indices.add(pred_chunk_idx)
            match_list.append({'gt_idx': [gt_record['source_idx']], 'gt': gt_record['raw'], 'norm_gt': gt_record['norm'], 'gt_category_type': gt_record['category'], 'gt_position': gt_record['position'], 'gt_attribute': gt_record['attribute'], 'pred_idx': pred_chunk['pred_indices'], 'pred': pred_chunk['raw'], 'norm_pred': pred_chunk['norm'], 'pred_category_type': pred_chunk['category'], 'pred_position': pred_chunk['position'], 'edit': pure_edit, 'img_id': img_name})
        else:
            append_unmatched_gt(gt_record)
    for (pred_chunk_idx, pred_chunk) in enumerate(pred_chunks):
        if pred_chunk_idx not in matched_chunk_indices:
            append_unmatched_pred(pred_chunk)
    return MatchResult(match_list, fallback_reason=fallback_reason)

def sort_by_position_skip_inline(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    先按 position[0] 从小到大排序；
    若 fine_category_type == 'equation_inline'，则统一放到最后，
    并保持它们在原列表中的相对顺序（稳定排序）。
    """
    return sorted(enumerate(items), key=lambda pair: (pair[1].get('fine_category_type') == 'equation_inline', pair[1]['position'][0], pair[0]))

def match_gt2pred_quick(gt_items, pred_items, line_type, img_name, truncated_timeout_sec=None, fallback_short_line_max_chars=None, fallback_target_chunk_chars=None, fallback_max_chunk_chars=None, fallback_max_chunk_span=8, fallback_order_window=None, fallback_order_penalty=0.08, split_pred_formula=True):
    pred_items = [pair[1] for pair in sort_by_position_skip_inline(pred_items)]
    (gt_lines, norm_gt_lines, gt_cat_list, pred_lines, norm_pred_lines, gt_items, pred_items) = get_gt_pred_lines(gt_items, pred_items, None)


    if not norm_gt_lines:
        match_list = []
        for pred_idx in range(len(norm_pred_lines)):
            match_list.append({'gt_idx': [''], 'gt': '', 'pred_idx': [pred_idx], 'pred': pred_lines[pred_idx], 'gt_position': [''], 'pred_position': pred_items[pred_idx]['position'][0], 'norm_gt': '', 'norm_pred': norm_pred_lines[pred_idx], 'gt_category_type': '', 'pred_category_type': get_pred_category_type(pred_idx, pred_items), 'gt_attribute': [{}], 'edit': 1, 'img_id': img_name})
        return match_list
    elif not norm_pred_lines:
        match_list = []
        for gt_idx in range(len(norm_gt_lines)):
            match_list.append({'gt_idx': [gt_idx], 'gt': gt_lines[gt_idx], 'pred_idx': [''], 'pred': '', 'gt_position': [gt_items[gt_idx].get('order') if gt_items[gt_idx].get('order') else gt_items[gt_idx].get('position', [''])[0]], 'pred_position': '', 'norm_gt': norm_gt_lines[gt_idx], 'norm_pred': '', 'gt_category_type': gt_cat_list[gt_idx], 'pred_category_type': '', 'gt_attribute': [gt_items[gt_idx].get('attribute', {})], 'edit': 1, 'img_id': img_name})
        return match_list
    elif len(norm_gt_lines) == 1 and len(norm_pred_lines) == 1:
        edit_distance = Levenshtein_distance(norm_gt_lines[0], norm_pred_lines[0])
        normalized_edit_distance = edit_distance / max(len(norm_gt_lines[0]), len(norm_pred_lines[0]))
        return [{'gt_idx': [0], 'gt': gt_lines[0], 'pred_idx': [0], 'pred': pred_lines[0], 'gt_position': [gt_items[0].get('order') if gt_items[0].get('order') else gt_items[0].get('position', [''])[0]], 'pred_position': pred_items[0]['position'][0], 'norm_gt': norm_gt_lines[0], 'norm_pred': norm_pred_lines[0], 'gt_category_type': gt_cat_list[0], 'pred_category_type': get_pred_category_type(0, pred_items), 'gt_attribute': [gt_items[0].get('attribute', {})], 'edit': normalized_edit_distance, 'img_id': img_name}]
    ignores = ['figure_caption', 'figure_footnote', 'table_caption', 'table_footnote', 'code_algorithm', 'code_algorithm_caption', 'header', 'footer', 'page_footnote', 'page_number', 'equation_caption']
    ignore_gt_lines = []
    ignores_ori_gt_lines = []
    ignores_gt_items = []
    ignore_gt_idxs = []
    ignores_gt_cat_list = []
    no_ignores_gt_lines = []
    no_ignores_ori_gt_lines = []
    no_ignores_gt_idxs = []
    no_ignores_gt_items = []
    no_ignores_gt_cat_list = []
    for (i, line) in enumerate(norm_gt_lines):
        if gt_cat_list[i] in ignores:
            ignore_gt_lines.append(line)
            ignores_ori_gt_lines.append(gt_lines[i])
            ignores_gt_items.append(gt_items[i])
            ignore_gt_idxs.append(i)
            ignores_gt_cat_list.append(gt_cat_list[i])
        else:
            no_ignores_gt_lines.append(line)
            no_ignores_ori_gt_lines.append(gt_lines[i])
            no_ignores_gt_items.append(gt_items[i])
            no_ignores_gt_cat_list.append(gt_cat_list[i])
            no_ignores_gt_idxs.append(i)
    ignore_pred_idxs = []
    ignore_pred_lines = []
    ignores_pred_items = []
    ignores_ori_pred_lines = []
    merged_ignore_results = []
    if len(ignore_gt_lines) > 0:
        ignore_matches_dict = {}
        ignore_matrix = compute_edit_distance_matrix_new(ignore_gt_lines, norm_pred_lines)
        ignores_gt_indices = set(range(len(ignore_gt_lines)))
        ignores_pred_indices = set(range(len(ignore_pred_lines)))
        ignore_matches = np.argwhere(ignore_matrix < 0.25)
        if len(ignore_matches) > 0:
            ignore_pred_idxs = [_[1] for _ in ignore_matches]

            for i in ignore_pred_idxs:
                ignore_pred_lines.append(norm_pred_lines[i])
                ignores_ori_pred_lines.append(pred_lines[i])
                ignores_pred_items.append(pred_items[i])
                ignores_gt_indices = set(range(len(ignore_gt_lines)))
                ignores_pred_indices = set(range(len(ignore_pred_lines)))
            for (idx, i) in enumerate(ignore_matches):
                ignore_matches_dict[i[0]] = {'pred_indices': [idx], 'edit_distance': ignore_matrix[i[0]][i[1]]}
        ignore_final_matches = merge_matches(ignore_matches_dict, {})
        recalculate_edit_distances(ignore_final_matches, {}, ignore_gt_lines, ignore_pred_lines)
        converted_ignore_results = convert_final_matches(ignore_final_matches, ignore_gt_lines, ignore_pred_lines)
        merged_ignore_results = merge_duplicates_add_unmatched(converted_ignore_results, ignore_gt_lines, ignore_pred_lines, ignores_ori_gt_lines, ignores_ori_pred_lines, ignores_gt_indices, ignores_pred_indices)
        for entry in merged_ignore_results:
            entry['gt_idx'] = [entry['gt_idx']] if not isinstance(entry['gt_idx'], list) else entry['gt_idx']
            entry['pred_idx'] = [entry['pred_idx']] if not isinstance(entry['pred_idx'], list) else entry['pred_idx']
            entry['gt_position'] = [ignores_gt_items[_].get('order') if ignores_gt_items[_].get('order') else ignores_gt_items[_].get('position', [''])[0] for _ in entry['gt_idx']] if entry['gt_idx'] != [''] else ['']
            entry['pred_position'] = ignores_pred_items[entry['pred_idx'][0]]['position'][0] if entry['pred_idx'] != [''] else ''
            entry['gt'] = ''.join([ignores_ori_gt_lines[_] for _ in entry['gt_idx']]) if entry['gt_idx'] != [''] else ''
            entry['pred'] = ''.join([ignores_ori_pred_lines[_] for _ in entry['pred_idx']]) if entry['pred_idx'] != [''] else ''
            entry['norm_gt'] = ''.join([ignore_gt_lines[_] for _ in entry['gt_idx']]) if entry['gt_idx'] != [''] else ''
            entry['norm_pred'] = ''.join([ignore_pred_lines[_] for _ in entry['pred_idx']]) if entry['pred_idx'] != [''] else ''
            if entry['gt_idx'] != ['']:
                ignore_type = ['figure_caption', 'figure_footnote', 'table_caption', 'table_footnote', 'code_algorithm', 'code_algorithm_caption', 'header', 'footer', 'page_footnote', 'page_number', 'equation_caption']
                gt_cagegory_clean = [ignores_gt_cat_list[_] for _ in entry['gt_idx'] if ignores_gt_cat_list[_] not in ignore_type]
                if gt_cagegory_clean:
                    entry['gt_category_type'] = Counter(gt_cagegory_clean).most_common(1)[0][0]
                else:
                    entry['gt_category_type'] = Counter([ignores_gt_cat_list[_] for _ in entry['gt_idx']]).most_common(1)[0][0]
            else:
                entry['gt_category_type'] = ''
                entry['pred_category_type'] = get_pred_category_type(entry['pred_idx'][0], ignores_pred_items) if entry['pred_idx'] != [''] else ''
                if entry['pred_category_type'] == 'equation_inline':
                    merged_ignore_results.remove(entry)
            entry['pred_category_type'] = get_pred_category_type(entry['pred_idx'][0], ignores_pred_items) if entry['pred_idx'] != [''] else ''
            entry['gt_attribute'] = [ignores_gt_items[_].get('attribute', {}) for _ in entry['gt_idx']] if entry['gt_idx'] != [''] else [{}]
            entry['img_id'] = img_name
        for entry in merged_ignore_results:
            if isinstance(entry['gt_idx'], list) and entry['gt_idx'] != ['']:
                gt_idx = []
                for i in entry['gt_idx']:
                    gt_idx.append(ignore_gt_idxs[i])
                entry['gt_idx'] = gt_idx
            if isinstance(entry['pred_idx'], list) and entry['pred_idx'] != ['']:
                pred_idx = []
                for i in entry['pred_idx']:
                    pred_idx.append(int(ignore_pred_idxs[i]))
                entry['pred_idx'] = pred_idx
    no_ignores_pred_lines = []
    no_ignores_ori_pred_lines = []
    no_ignores_pred_indices = []
    no_ignores_pred_items = []
    no_ignore_pred_idxs = []
    for (idx, line) in enumerate(norm_pred_lines):
        if not idx in ignore_pred_idxs:
            no_ignores_pred_lines.append(line)
            no_ignores_ori_pred_lines.append(pred_lines[idx])
            no_ignores_pred_items.append(pred_items[idx])
            no_ignore_pred_idxs.append(idx)
    no_ignores_gt_indices = set(range(len(no_ignores_gt_lines)))
    no_ignores_pred_indices = set(range(len(no_ignores_pred_lines)))
    cost_matrix = compute_edit_distance_matrix_new(no_ignores_gt_lines, no_ignores_pred_lines)
    try:
        (matched_col_idx, row_ind, cost_list) = cal_final_match(cost_matrix, no_ignores_gt_lines, no_ignores_pred_lines, truncated_timeout_sec=truncated_timeout_sec)
    except TruncatedMatchTimeout:
        print(f'[quick-match-timeout] {img_name}: deal_with_truncated exceeded {truncated_timeout_sec}s, fallback to chunked Hungarian', flush=True)
        return match_gt2pred_timeout_safe(gt_items, pred_items, img_name, short_line_max_chars=fallback_short_line_max_chars, target_chunk_chars=fallback_target_chunk_chars, max_chunk_chars=fallback_max_chunk_chars, max_chunk_span=fallback_max_chunk_span, order_window=fallback_order_window, order_penalty=fallback_order_penalty, fallback_reason='quick_match_timeout')
    (gt_lens_dict, pred_lens_dict) = initialize_indices(no_ignores_gt_lines, no_ignores_pred_lines)
    (matches, unmatched_gt_indices, unmatched_pred_indices) = process_matches(matched_col_idx, row_ind, cost_list, no_ignores_gt_lines, no_ignores_pred_lines, no_ignores_ori_pred_lines)
    matching_dict = fuzzy_match_unmatched_items(unmatched_gt_indices, no_ignores_gt_lines, no_ignores_pred_lines, gt_cat_list=no_ignores_gt_cat_list)
    final_matches = merge_matches(matches, matching_dict)
    recalculate_edit_distances(final_matches, gt_lens_dict, no_ignores_gt_lines, no_ignores_pred_lines)
    converted_results = convert_final_matches(final_matches, no_ignores_gt_lines, no_ignores_pred_lines)
    merged_results = merge_duplicates_add_unmatched(converted_results, no_ignores_gt_lines, no_ignores_pred_lines, no_ignores_ori_gt_lines, no_ignores_ori_pred_lines, no_ignores_gt_indices, no_ignores_pred_indices)
    for entry in merged_results:
        if entry['gt_idx'] != ['']:
            ignore_type = ['figure_caption', 'figure_footnote', 'table_caption', 'table_footnote', 'code_algorithm', 'code_algorithm_caption', 'header', 'footer', 'page_footnote', 'page_number', 'equation_caption']
            gt_cagegory_clean = [no_ignores_gt_cat_list[_] for _ in entry['gt_idx'] if no_ignores_gt_cat_list[_] not in ignore_type]
            if gt_cagegory_clean:
                entry['gt_category_type'] = Counter(gt_cagegory_clean).most_common(1)[0][0]
            else:
                entry['gt_category_type'] = Counter([no_ignores_gt_cat_list[_] for _ in entry['gt_idx']]).most_common(1)[0][0]
        else:
            entry['gt_category_type'] = ''
            entry['pred_category_type'] = get_pred_category_type(entry['pred_idx'][0], no_ignores_pred_items) if entry['pred_idx'] != [''] else ''
            if entry['pred_category_type'] == 'equation_inline':
                merged_results.remove(entry)
        entry['gt_idx'] = [entry['gt_idx']] if not isinstance(entry['gt_idx'], list) else entry['gt_idx']
        entry['pred_idx'] = [entry['pred_idx']] if not isinstance(entry['pred_idx'], list) else entry['pred_idx']
        entry['gt_position'] = [no_ignores_gt_items[_].get('order') if no_ignores_gt_items[_].get('order') else no_ignores_gt_items[_].get('position', [''])[0] for _ in entry['gt_idx']] if entry['gt_idx'] != [''] else ['']
        entry['pred_position'] = no_ignores_pred_items[entry['pred_idx'][0]]['position'][0] if entry['pred_idx'] != [''] else ''
        if entry['gt_category_type'] == 'equation_isolated' and len(entry['gt_idx']) > 1:
            mutli_formula = ' \\\\ '.join(['{' + strip_formula_delimiters(no_ignores_ori_gt_lines[_]) + '}' for _ in entry['gt_idx']]) if entry['gt_idx'] != [''] else ''
            mutli_formula = '\\begin{array}{l} ' + mutli_formula + ' \\end{array}'
            entry['gt'] = mutli_formula
        else:
            entry['gt'] = ''.join([no_ignores_ori_gt_lines[_] for _ in entry['gt_idx']]) if entry['gt_idx'] != [''] else ''
        entry['pred_category_type'] = get_pred_category_type(entry['pred_idx'][0], no_ignores_pred_items) if entry['pred_idx'] != [''] else ''
        entry['gt_attribute'] = [no_ignores_gt_items[_].get('attribute', {}) for _ in entry['gt_idx']] if entry['gt_idx'] != [''] else [{}]
        entry['img_id'] = img_name
        if 'equation' in entry['pred_category_type'] and len(entry['pred_idx']) > 1:
            mutli_formula = ' \\\\ '.join(['{' + strip_formula_delimiters(no_ignores_ori_pred_lines[_]) + '}' for _ in entry['pred_idx']]) if entry['pred_idx'] != [''] else ''
            mutli_formula = '\\begin{array}{l} ' + mutli_formula + ' \\end{array}'
            entry['pred'] = mutli_formula
        else:
            entry['pred'] = ''.join([no_ignores_ori_pred_lines[_] for _ in entry['pred_idx']]) if entry['pred_idx'] != [''] else ''
        entry['norm_gt'] = ''.join([no_ignores_gt_lines[_] for _ in entry['gt_idx']]) if entry['gt_idx'] != [''] else ''
        entry['norm_pred'] = ''.join([no_ignores_pred_lines[_] for _ in entry['pred_idx']]) if entry['pred_idx'] != [''] else ''
    for entry in merged_results:
        if isinstance(entry['gt_idx'], list) and entry['gt_idx'] != ['']:
            gt_idx = []
            for i in entry['gt_idx']:
                gt_idx.append(no_ignores_gt_idxs[i])
            entry['gt_idx'] = gt_idx
        if isinstance(entry['pred_idx'], list) and entry['pred_idx'] != ['']:
            pred_idx = []
            for i in entry['pred_idx']:
                pred_idx.append(int(no_ignore_pred_idxs[i]))
            entry['pred_idx'] = pred_idx
    if len(merged_ignore_results) > 0:
        merged_results.extend(merged_ignore_results)
    return merged_results

def merge_duplicates_add_unmatched(converted_results, norm_gt_lines, norm_pred_lines, gt_lines, pred_lines, all_gt_indices, all_pred_indices):
    merged_results = []
    processed_pred = set()
    processed_gt = set()
    for entry in converted_results:
        pred_idx = tuple(entry['pred_idx']) if isinstance(entry['pred_idx'], list) else (entry['pred_idx'],)
        if pred_idx not in processed_pred and pred_idx != ('',):
            merged_entry = {'gt_idx': [entry['gt_idx']], 'gt': entry['gt'], 'pred_idx': entry['pred_idx'], 'pred': entry['pred'], 'edit': entry['edit']}
            for other_entry in converted_results:
                other_pred_idx = tuple(other_entry['pred_idx']) if isinstance(other_entry['pred_idx'], list) else (other_entry['pred_idx'],)
                if other_pred_idx == pred_idx and other_entry is not entry:
                    merged_entry['gt_idx'].append(other_entry['gt_idx'])
                    merged_entry['gt'] += other_entry['gt']
                    processed_gt.add(other_entry['gt_idx'])
            merged_results.append(merged_entry)
            processed_pred.add(pred_idx)
            processed_gt.add(entry['gt_idx'])
    for gt_idx in range(len(norm_gt_lines)):
        if gt_idx not in processed_gt:
            merged_results.append({'gt_idx': [gt_idx], 'gt': gt_lines[gt_idx], 'pred_idx': [''], 'pred': '', 'edit': 1})
    return merged_results

def merge_lists_with_sublists(main_list, sub_lists):
    main_list_final = list(copy.deepcopy(main_list))
    for sub_list in sub_lists:
        pop_idx = main_list_final.index(sub_list[0])
        for _ in sub_list:
            main_list_final.pop(pop_idx)
        main_list_final.insert(pop_idx, sub_list)
    return main_list_final

def sub_pred_fuzzy_matching(gt, pred):
    min_d = float('inf')
    gt_len = len(gt)
    pred_len = len(pred)
    if gt_len >= pred_len and pred_len > 0:
        for i in range(gt_len - pred_len + 1):
            sub = gt[i:i + pred_len]
            dist = Levenshtein_distance(sub, pred) / pred_len
            if dist < min_d:
                min_d = dist

        return min_d
    else:
        return False

def sub_gt_fuzzy_matching(pred, gt):
    min_d = float('inf')
    pos = ''
    matched_sub = ''
    gt_len = len(gt)
    pred_len = len(pred)
    if pred_len >= gt_len and gt_len > 0:
        for i in range(pred_len - gt_len + 1):
            sub = pred[i:i + gt_len]
            dist = Levenshtein.distance(sub, gt) / gt_len
            if dist < min_d:
                min_d = dist
                pos = i
                matched_sub = sub
        return (min_d, pos, gt_len, matched_sub)
    else:
        return (1, '', gt_len, '')

def get_final_subset(subset_certain, subset_certain_cost):
    if not subset_certain or not subset_certain_cost:
        return []
    subset_turple = sorted([(a, b) for (a, b) in zip(subset_certain, subset_certain_cost)], key=lambda x: x[0][0])
    group_list = defaultdict(list)
    group_idx = 0
    group_list[group_idx].append(subset_turple[0])
    for item in subset_turple[1:]:
        overlap_flag = False
        for subset in group_list[group_idx]:
            for idx in item[0]:
                if idx in subset[0]:
                    overlap_flag = True
                    break
            if overlap_flag:
                break
        if overlap_flag:
            group_list[group_idx].append(item)
        else:
            group_idx += 1
            group_list[group_idx].append(item)
    final_subset = []
    for (_, group) in group_list.items():
        if len(group) == 1:
            final_subset.append(group[0][0])
        else:
            path_dict = defaultdict(list)
            path_idx = 0
            path_dict[path_idx].append(group[0])
            for subset in group[1:]:
                new_path = True
                for (path_idx_s, path_items) in path_dict.items():
                    is_dup = False
                    is_same = False
                    for path_item in path_items:
                        if path_item[0] == subset[0]:
                            is_dup = True
                            is_same = True
                            if path_item[1] > subset[1]:
                                path_dict[path_idx_s].pop(path_dict[path_idx_s].index(path_item))
                                path_dict[path_idx_s].append(subset)
                        else:
                            for num_1 in path_item[0]:
                                for num_2 in subset[0]:
                                    if num_1 == num_2:
                                        is_dup = True
                    if not is_dup:
                        path_dict[path_idx_s].append(subset)
                        new_path = False
                    if is_same:
                        new_path = False
                if new_path:
                    path_idx = len(path_dict.keys())
                    path_dict[path_idx].append(subset)
            saved_cost = float('inf')
            saved_subset = []
            for (path_idx, path) in path_dict.items():
                avg_cost = sum([i[1] for i in path]) / len(path)
                if avg_cost < saved_cost:
                    saved_subset = [i[0] for i in path]
                    saved_cost = avg_cost
            final_subset.extend(saved_subset)
    return final_subset

def judge_pred_merge(gt_list, pred_list, threshold=0.6):
    if len(pred_list) == 1:
        return (False, False)
    cur_pred = ' '.join(pred_list[:-1])
    merged_pred = ' '.join(pred_list)
    cur_dist = Levenshtein.distance(gt_list[0], cur_pred) / max(len(gt_list[0]), len(cur_pred))
    merged_dist = Levenshtein.distance(gt_list[0], merged_pred) / max(len(gt_list[0]), len(merged_pred))
    if merged_dist > cur_dist:
        return (False, False)
    cur_fuzzy_dists = [sub_pred_fuzzy_matching(gt_list[0], cur_pred) for cur_pred in pred_list[:-1]]
    if any((dist is False or dist > threshold for dist in cur_fuzzy_dists)):
        return (False, False)
    add_fuzzy_dist = sub_pred_fuzzy_matching(gt_list[0], pred_list[-1])
    if add_fuzzy_dist is False:
        return (False, False)
    merged_pred_flag = add_fuzzy_dist < threshold
    continue_flag = len(merged_pred) <= len(gt_list[0])
    return (merged_pred_flag, continue_flag)

def deal_with_truncated(cost_matrix, norm_gt_lines, norm_pred_lines, deadline=None):
    matched_first = np.argwhere(cost_matrix < 0.25)
    masked_gt_idx = [i[0] for i in matched_first]
    unmasked_gt_idx = [i for i in range(cost_matrix.shape[0]) if i not in masked_gt_idx]
    masked_pred_idx = [i[1] for i in matched_first]
    unmasked_pred_idx = [i for i in range(cost_matrix.shape[1]) if i not in masked_pred_idx]
    merges_gt_dict = {}


    for gt_idx in unmasked_gt_idx:
        if deadline is not None and time.monotonic() >= deadline:
            raise TruncatedMatchTimeout()
        check_merge_subset = []
        merged_dist = []
        for pred_idx in unmasked_pred_idx:
            if deadline is not None and time.monotonic() >= deadline:
                raise TruncatedMatchTimeout()
            step = 1
            merged_pred = [norm_pred_lines[pred_idx]]
            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TruncatedMatchTimeout()
                if step >= MAX_TRUNCATED_PRED_MERGE:
                    break
                if pred_idx + step in masked_pred_idx or pred_idx + step >= len(norm_pred_lines):
                    break
                else:
                    merged_pred.append(norm_pred_lines[pred_idx + step])
                    (merged_pred_flag, continue_flag) = judge_pred_merge([norm_gt_lines[gt_idx]], merged_pred)
                    if not merged_pred_flag:
                        break
                    else:
                        step += 1
                    if not continue_flag:
                        break
            check_merge_subset.append(list(range(pred_idx, pred_idx + step)))
            matched_line = ' '.join([norm_pred_lines[i] for i in range(pred_idx, pred_idx + step)])
            dist = Levenshtein_distance(norm_gt_lines[gt_idx], matched_line) / max(len(matched_line), len(norm_gt_lines[gt_idx]))
            merged_dist.append(dist)
        if not merged_dist:
            subset_certain = []
            min_cost_idx = ''
            min_cost = float('inf')
        else:
            min_cost = min(merged_dist)
            min_cost_idx = merged_dist.index(min_cost)
            subset_certain = check_merge_subset[min_cost_idx]
        merges_gt_dict[gt_idx] = {'merge_subset': check_merge_subset, 'merged_cost': merged_dist, 'min_cost_idx': min_cost_idx, 'subset_certain': subset_certain, 'min_cost': min_cost}
    subset_certain = [merges_gt_dict[gt_idx]['subset_certain'] for gt_idx in unmasked_gt_idx if merges_gt_dict[gt_idx]['subset_certain']]
    subset_certain_cost = [merges_gt_dict[gt_idx]['min_cost'] for gt_idx in unmasked_gt_idx if merges_gt_dict[gt_idx]['subset_certain']]
    subset_certain_final = get_final_subset(subset_certain, subset_certain_cost)
    if not subset_certain_final:
        return (cost_matrix, norm_pred_lines, range(len(norm_pred_lines)))
    final_pred_idx_list = merge_lists_with_sublists(range(len(norm_pred_lines)), subset_certain_final)
    final_norm_pred_lines = [' '.join(norm_pred_lines[idx_list[0]:idx_list[-1] + 1]) if isinstance(idx_list, list) else norm_pred_lines[idx_list] for idx_list in final_pred_idx_list]
    new_cost_matrix = compute_edit_distance_matrix_new(norm_gt_lines, final_norm_pred_lines)
    return (new_cost_matrix, final_norm_pred_lines, final_pred_idx_list)

def cal_final_match(cost_matrix, norm_gt_lines, norm_pred_lines, truncated_timeout_sec=None):
    deadline = None
    if truncated_timeout_sec is not None:
        try:
            truncated_timeout_sec = float(truncated_timeout_sec)
        except (TypeError, ValueError):
            truncated_timeout_sec = None
        if truncated_timeout_sec is not None and truncated_timeout_sec > 0:
            deadline = time.monotonic() + truncated_timeout_sec
    (new_cost_matrix, final_norm_pred_lines, final_pred_idx_list) = deal_with_truncated(cost_matrix, norm_gt_lines, norm_pred_lines, deadline=deadline)
    (row_ind, col_ind) = linear_sum_assignment(new_cost_matrix)
    cost_list = [new_cost_matrix[r][c] for (r, c) in zip(row_ind, col_ind)]
    matched_col_idx = [final_pred_idx_list[i] for i in col_ind]
    return (matched_col_idx, row_ind, cost_list)

def initialize_indices(norm_gt_lines, norm_pred_lines):
    gt_lens_dict = {idx: len(gt_line) for (idx, gt_line) in enumerate(norm_gt_lines)}
    pred_lens_dict = {idx: len(pred_line) for (idx, pred_line) in enumerate(norm_pred_lines)}
    return (gt_lens_dict, pred_lens_dict)

def process_matches(matched_col_idx, row_ind, cost_list, norm_gt_lines, norm_pred_lines, pred_lines):
    matches = {}
    unmatched_gt_indices = []
    unmatched_pred_indices = []
    for i in range(len(norm_gt_lines)):
        if i in row_ind:
            idx = list(row_ind).index(i)
            pred_idx = matched_col_idx[idx]
            if pred_idx is None or (isinstance(pred_idx, list) and None in pred_idx):
                unmatched_pred_indices.append(pred_idx)
                continue
            if isinstance(pred_idx, list):


                matched_pred_indices_range = list(range(pred_idx[0], pred_idx[-1] + 1))
            else:


                matched_pred_indices_range = [pred_idx]
            edit = cost_list[idx]
            if edit > 0.7:
                unmatched_pred_indices.extend(matched_pred_indices_range)
                unmatched_gt_indices.append(i)
            else:
                matches[i] = {'pred_indices': matched_pred_indices_range, 'edit_distance': edit}
                for matched_pred_idx in matched_pred_indices_range:
                    if matched_pred_idx in unmatched_pred_indices:
                        unmatched_pred_indices.remove(matched_pred_idx)
        else:
            unmatched_gt_indices.append(i)
    return (matches, unmatched_gt_indices, unmatched_pred_indices)

def fuzzy_match_unmatched_items(unmatched_gt_indices, norm_gt_lines, norm_pred_lines, gt_cat_list=None):
    matching_dict = {}
    for (pred_idx, pred_content) in enumerate(norm_pred_lines):
        if isinstance(pred_idx, list):
            continue
        matching_indices = []
        for unmatched_gt_idx in unmatched_gt_indices:
            if gt_cat_list and gt_cat_list[unmatched_gt_idx] == 'equation_isolated':
                continue
            gt_content = norm_gt_lines[unmatched_gt_idx]
            (cur_fuzzy_dist_unmatch, cur_pos, gt_lens, matched_field) = sub_gt_fuzzy_matching(pred_content, gt_content)
            if cur_fuzzy_dist_unmatch < 0.4:
                matching_indices.append(unmatched_gt_idx)
        if matching_indices:
            matching_dict[pred_idx] = matching_indices
    return matching_dict

def merge_matches(matches, matching_dict):
    final_matches = {}
    processed_gt_indices = set()
    for (gt_idx, match_info) in matches.items():
        pred_indices = match_info['pred_indices']
        edit_distance = match_info['edit_distance']
        pred_key = tuple(sorted(pred_indices))
        if pred_key in final_matches:
            if gt_idx not in processed_gt_indices:
                final_matches[pred_key]['gt_indices'].append(gt_idx)
                processed_gt_indices.add(gt_idx)
        else:
            final_matches[pred_key] = {'gt_indices': [gt_idx], 'edit_distance': edit_distance}
            processed_gt_indices.add(gt_idx)
    for (pred_idx, gt_indices) in matching_dict.items():
        pred_key = (pred_idx,) if not isinstance(pred_idx, (list, tuple)) else tuple(sorted(pred_idx))
        if pred_key in final_matches:
            for gt_idx in gt_indices:
                if gt_idx not in processed_gt_indices:
                    final_matches[pred_key]['gt_indices'].append(gt_idx)
                    processed_gt_indices.add(gt_idx)
        else:
            final_matches[pred_key] = {'gt_indices': [gt_idx for gt_idx in gt_indices if gt_idx not in processed_gt_indices], 'edit_distance': None}
            processed_gt_indices.update(final_matches[pred_key]['gt_indices'])
    return final_matches

def recalculate_edit_distances(final_matches, gt_lens_dict, norm_gt_lines, norm_pred_lines):
    for (pred_key, info) in final_matches.items():
        gt_indices = sorted(set(info['gt_indices']))
        if not gt_indices:
            info['edit_distance'] = 1
            continue
        if len(gt_indices) > 1:
            merged_gt_content = ''.join((norm_gt_lines[gt_idx] for gt_idx in gt_indices))
            pred_content = norm_pred_lines[pred_key[0]] if isinstance(pred_key[0], int) else ''
            try:
                edit_distance = Levenshtein_distance(merged_gt_content, pred_content)
                normalized_edit_distance = edit_distance / max(len(merged_gt_content), len(pred_content))
            except ZeroDivisionError:
                normalized_edit_distance = 1
            info['edit_distance'] = normalized_edit_distance
        else:
            gt_idx = gt_indices[0]
            pred_content = ' '.join((norm_pred_lines[pred_idx] for pred_idx in pred_key if isinstance(pred_idx, int)))
            try:
                edit_distance = Levenshtein_distance(norm_gt_lines[gt_idx], pred_content)
                normalized_edit_distance = edit_distance / max(len(norm_gt_lines[gt_idx]), len(pred_content))
            except ZeroDivisionError:
                normalized_edit_distance = 1
            info['edit_distance'] = normalized_edit_distance
            info['pred_content'] = pred_content

def convert_final_matches(final_matches, norm_gt_lines, norm_pred_lines):
    converted_results = []
    all_gt_indices = set(range(len(norm_gt_lines)))
    all_pred_indices = set(range(len(norm_pred_lines)))
    for (pred_key, info) in final_matches.items():
        pred_content = ' '.join((norm_pred_lines[pred_idx] for pred_idx in pred_key if isinstance(pred_idx, int)))
        for gt_idx in sorted(set(info['gt_indices'])):
            result_entry = {'gt_idx': int(gt_idx), 'gt': norm_gt_lines[gt_idx], 'pred_idx': list(pred_key), 'pred': pred_content, 'edit': info['edit_distance']}
            converted_results.append(result_entry)
    matched_gt_indices = set().union(*[set(info['gt_indices']) for info in final_matches.values()])
    unmatched_gt_indices = all_gt_indices - matched_gt_indices
    matched_pred_indices = set((idx for pred_key in final_matches.keys() for idx in pred_key if isinstance(idx, int)))
    unmatched_pred_indices = all_pred_indices - matched_pred_indices
    if unmatched_pred_indices:
        if unmatched_gt_indices:
            distance_matrix = [[Levenshtein_distance(norm_gt_lines[gt_idx], norm_pred_lines[pred_idx]) / max(len(norm_gt_lines[gt_idx]), len(norm_pred_lines[pred_idx])) for pred_idx in unmatched_pred_indices] for gt_idx in unmatched_gt_indices]
            (row_ind, col_ind) = linear_sum_assignment(distance_matrix)
            for (i, j) in zip(row_ind, col_ind):
                gt_idx = list(unmatched_gt_indices)[i]
                pred_idx = list(unmatched_pred_indices)[j]
                result_entry = {'gt_idx': int(gt_idx), 'gt': norm_gt_lines[gt_idx], 'pred_idx': [pred_idx], 'pred': norm_pred_lines[pred_idx], 'edit': 1}
                converted_results.append(result_entry)
            matched_gt_indices.update((list(unmatched_gt_indices)[i] for i in row_ind))
        else:
            result_entry = {'gt_idx': '', 'gt': '', 'pred_idx': list(unmatched_pred_indices), 'pred': ' '.join((norm_pred_lines[pred_idx] for pred_idx in unmatched_pred_indices)), 'edit': 1}
            converted_results.append(result_entry)
    else:
        for gt_idx in unmatched_gt_indices:
            result_entry = {'gt_idx': int(gt_idx), 'gt': norm_gt_lines[gt_idx], 'pred_idx': '', 'pred': '', 'edit': 1}
            converted_results.append(result_entry)
    return converted_results
