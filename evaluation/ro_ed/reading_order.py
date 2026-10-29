# Copyright (c) OpenDataLab. Adapted for TongGuOCR under Apache-2.0.
# Source: OmniDocBench be7a44d371e43801fabd2e31e02976f106b6aa16.
# Text-only matching path; see THIRD_PARTY_NOTICES.md.
"""Local span repair and GT-order pairing for text-only RO-ED."""
from __future__ import annotations
import logging
import re
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from pathlib import Path
import hashlib
import json
import numpy as np
import Levenshtein
from func_timeout import FunctionTimedOut, func_timeout
from .matching import match_gt2pred_quick, match_gt2pred_timeout_safe
from .protocol import preprocess_gt, preprocess_prediction, protocol_spec
from .normalize import normalized_text, textblock2unicode, textblock_with_norm_formula
logger = logging.getLogger(__name__)


def score_page(record):
    """Match line content, then score the sequence of matched GT order IDs."""
    page_id = str(record['page_id'])
    gt, gt_stats = preprocess_gt(str(record.get('label') or ''))
    pred, pred_stats = preprocess_prediction(str(record.get('predict') or ''))
    gt_items = [dict(category_type='text_block', text=text, order=i, attribute={})
                for i, text in enumerate(gt, 1)]
    offset = 0
    pred_items = []
    for text in pred:
        pred_items.append(dict(category_type='text_all', fine_category_type='text_block',
                               content=text, position=[offset, offset + len(text)]))
        offset += len(text) + 1
    kwargs = dict(truncated_timeout_sec=300, fallback_max_chunk_span=10, fallback_order_penalty=0.10)
    try:
        matches = func_timeout(420, match_gt2pred_quick,
                               args=(gt_items, pred_items, 'text_all', page_id + '.jpg'), kwargs=kwargs)
    except FunctionTimedOut:
        matches = match_gt2pred_timeout_safe(gt_items, pred_items, page_id + '.jpg',
                                             max_chunk_span=10, order_penalty=0.10, fallback_reason='page_timeout')
    fallback = getattr(matches, 'fallback_reason', None)
    engine = TextLineMatcher()
    matches = engine._maybe_apply_local_text_span_fallback(matches, pred_items)
    # Unmatched predictions have no GT order ID in the archived protocol.
    matches = [m for m in matches if m.get('gt_category_type') == 'text_block']
    paired = engine.get_order_paired(matches, page_id + '.jpg') if matches else {}
    return dict(page_id=page_id, gt=paired.get('gt', []), pred=paired.get('pred', []),
                edit=paired.get('edit'), fallback=fallback,
                empty_gt=not gt, empty_pred=not pred, gt_stats=gt_stats.to_dict(), pred_stats=pred_stats.to_dict())


def calculate_reading_order(records, workers=12, *, source_path=None, output_dir=None):
    """Evaluate token-form {page_id,predict,label} rows without OmniDocBench.

    Empty GT/prediction pairs have no order sequence and are excluded from
    the mean, matching the archived scorer. A missing prediction with GT
    incurs deletions. This is a page mean, not a corpus character distance.
    """
    if workers < 1:
        raise ValueError('workers must be positive')
    ids = [str(r['page_id']) for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate page IDs in reading-order input')
    if not records:
        raise ValueError('reading-order input is empty')
    if workers == 1:
        pages = list(map(score_page, records))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pages = list(pool.map(score_page, records, chunksize=4))
    scored = sorted((p for p in pages if p['edit'] is not None), key=lambda p: p['page_id'])
    upper = sum(max(len(p['gt']), len(p['pred'])) for p in scored)
    edits = sum(Levenshtein.distance(p['gt'], p['pred']) for p in scored)
    fallbacks = Counter(p['fallback'] for p in pages if p['fallback'])
    summary = dict(status='ok', engine='tongguocr_standalone', source_kind='predict_jsonl',
                   source_path=str(Path(source_path).resolve()) if source_path else None,
                   record_count=len(pages), official_page_count=len(scored), reading_order_sample_count=len(scored),
                   empty_gt_pages=sum(p['empty_gt'] for p in pages), empty_pred_pages=sum(p['empty_pred'] for p in pages),
                   ALL_page_avg=float(np.mean([p['edit'] for p in scored])) if scored else None,
                   edit_whole=edits / upper if upper else None,
                   edit_sample_avg=float(np.mean([p['edit'] for p in scored])) if scored else None,
                   quick_match_timeout_count=fallbacks['quick_match_timeout'], page_timeout_count=fallbacks['page_timeout'],
                   fallback_total=sum(fallbacks.values()),
                   protocol_sha256=hashlib.sha256(json.dumps(protocol_spec(), ensure_ascii=False, sort_keys=True).encode()).hexdigest())
    for prefix in ('gt', 'pred'):
        stats = Counter()
        for page in pages:
            stats.update({k: v for k, v in page[prefix + '_stats'].items() if isinstance(v, int)})
        summary.update({prefix + '_' + k: v for k, v in stats.items()})
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, value in [('summary.json', summary), ('per_page.json', pages), ('protocol.json', dict(spec=protocol_spec()))]:
            (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return summary

def sort_by_position_skip_inline(items):
    return sorted(enumerate(items), key=lambda pair: (pair[1].get('fine_category_type') == 'equation_inline', pair[1]['position'][0], pair[0]))

class TextLineMatcher:

    def _build_local_text_search_pool(self, pred_dataset_mix):
        pred_pool = []
        for (sorted_idx, (_, pred_item)) in enumerate(sort_by_position_skip_inline(pred_dataset_mix)):
            category_type = pred_item.get('category_type')
            if category_type not in ['text_all', 'equation_isolated']:
                continue
            raw_text = str(pred_item.get('content', '') or '')
            rendered_text = self._cross_category_text(category_type, raw_text)
            pred_pool.append({'sorted_idx': sorted_idx, 'category_type': category_type, 'raw_text': raw_text, 'rendered_text': rendered_text, 'pred_item': pred_item})
        return pred_pool

    def _cross_category_text(self, category_type, text):
        return textblock2unicode(str(text or ''))

    def _is_formula_category(self, category_type):
        return category_type in ['equation_isolated', 'equation_inline']

    def _match_item_pred_pool_indices(self, match_item, pred_pool):
        pred_indices = []
        for pred_idx in match_item.get('pred_idx', []) or []:
            if isinstance(pred_idx, np.integer):
                pred_idx = int(pred_idx)
            if isinstance(pred_idx, int) and 0 <= pred_idx < len(pred_pool):
                pred_indices.append(pred_idx)
        if pred_indices:
            return sorted(set(pred_indices))
        pred_position = match_item.get('pred_position')
        if pred_position in ([''], '', None):
            return []
        matched_indices = []
        for (idx, pool_item) in enumerate(pred_pool):
            position = pool_item['pred_item'].get('position', ['', ''])
            if not isinstance(position, list) or len(position) < 2:
                continue
            try:
                start = float(position[0])
                end = float(position[1])
                pred_pos = float(pred_position)
            except (TypeError, ValueError):
                continue
            if start <= pred_pos <= end:
                matched_indices.append(idx)
        return matched_indices

    def _maybe_apply_local_text_span_fallback(self, match, pred_dataset_mix):
        if not match:
            return match
        pred_pool = self._build_local_text_search_pool(pred_dataset_mix)
        if not pred_pool:
            return match
        updated_matches = [dict(item) for item in match]
        replacement_candidates = []
        for (match_idx, match_item) in enumerate(updated_matches):
            if not self._should_try_local_text_span_fallback(match_item):
                continue
            blocked_indices = set()
            for (other_idx, other_item) in enumerate(updated_matches):
                if other_idx == match_idx or other_item.get('gt_idx') == ['']:
                    continue
                blocked_indices.update(self._match_item_pred_pool_indices(other_item, pred_pool))
            candidate = self._search_best_local_text_span(match_item, pred_pool, blocked_indices)
            if not candidate:
                continue
            replacement_candidates.append({'match_idx': match_idx, 'gain': float(match_item.get('edit', 1)) - candidate['edit'], 'candidate': candidate})
        used_pred_indices = set()
        for replacement in sorted(replacement_candidates, key=lambda item: item['gain'], reverse=True):
            candidate = replacement['candidate']
            candidate_idx_set = set(candidate['pred_idx'])
            if candidate_idx_set & used_pred_indices:
                continue
            updated_matches[replacement['match_idx']].update({'pred_idx': candidate['pred_idx'], 'pred': candidate['pred'], 'pred_position': candidate['pred_position'], 'pred_category_type': candidate['pred_category_type'], 'norm_pred': candidate['norm_pred'], 'edit': candidate['edit']})
            used_pred_indices.update(candidate_idx_set)
            logger.info(f"[text-fallback] {updated_matches[replacement['match_idx']].get('img_id', '')}: gt_idx={updated_matches[replacement['match_idx']].get('gt_idx')} edit={float(replacement['gain']) + float(candidate['edit']):.4f}->{candidate['edit']:.4f} pred_idx={candidate['pred_idx']}")
        return updated_matches

    def _normalize_text_match_target(self, gt_category, gt_text, has_formula_hint=False):
        if has_formula_hint and (not self._is_formula_category(gt_category)):
            return normalized_text(textblock_with_norm_formula(str(gt_text or '')))
        return normalized_text(self._cross_category_text(gt_category, gt_text))

    def _normalized_edit_distance(self, gt_text, pred_text):
        if not gt_text and (not pred_text):
            return 0
        if not gt_text or not pred_text:
            return 1
        return Levenshtein.distance(gt_text, pred_text) / max(len(gt_text), len(pred_text))

    def _search_best_local_text_span(self, match_item, pred_pool, blocked_indices):
        if not pred_pool:
            return None
        gt_text = str(match_item.get('gt', '') or '')
        gt_category = match_item.get('gt_category_type', '')
        has_formula_hint = self._text_contains_formula_hint(gt_text)
        gt_norm = self._normalize_text_match_target(gt_category, gt_text, has_formula_hint=has_formula_hint)
        if not gt_norm:
            return None
        try:
            current_edit = float(match_item.get('edit', 1))
        except Exception:
            current_edit = 1.0
        owned_indices = self._match_item_pred_pool_indices(match_item, pred_pool)
        if owned_indices:
            lower_bound = max(0, min(owned_indices) - 8)
            upper_bound = min(len(pred_pool) - 1, max(owned_indices) + 8)
        else:
            lower_bound = 0
            upper_bound = len(pred_pool) - 1
        allowed_categories = {'text_all'}
        if has_formula_hint:
            allowed_categories.add('equation_isolated')
        max_span = 8 if has_formula_hint else 4
        max_allowed_edit = 0.45 if has_formula_hint else 0.35
        best_candidate = None
        for start_idx in range(lower_bound, upper_bound + 1):
            start_item = pred_pool[start_idx]
            if start_item['category_type'] not in allowed_categories or start_idx in blocked_indices:
                continue
            raw_parts = []
            rendered_parts = []
            for end_idx in range(start_idx, min(upper_bound + 1, start_idx + max_span)):
                current_item = pred_pool[end_idx]
                if current_item['category_type'] not in allowed_categories:
                    break
                if end_idx in blocked_indices:
                    break
                raw_parts.append(current_item['raw_text'])
                rendered_parts.append(current_item['rendered_text'])
                norm_pred = normalized_text(''.join(rendered_parts))
                if not norm_pred:
                    continue
                edit = self._normalized_edit_distance(gt_norm, norm_pred)
                candidate = {'pred_idx': list(range(start_idx, end_idx + 1)), 'pred': ''.join(raw_parts), 'pred_position': pred_pool[start_idx]['pred_item']['position'][0], 'pred_category_type': 'text_block', 'norm_pred': norm_pred, 'edit': edit, 'span_len': end_idx - start_idx + 1, 'distance_to_current': abs(start_idx - owned_indices[0]) if owned_indices else 0}
                if best_candidate is None:
                    best_candidate = candidate
                    continue
                if edit + 1e-12 < best_candidate['edit']:
                    best_candidate = candidate
                    continue
                if edit > best_candidate['edit'] + 1e-12:
                    continue
                if candidate['span_len'] < best_candidate['span_len']:
                    best_candidate = candidate
                    continue
                if candidate['span_len'] == best_candidate['span_len'] and candidate['distance_to_current'] < best_candidate['distance_to_current']:
                    best_candidate = candidate
        if not best_candidate:
            return None
        if best_candidate['edit'] >= current_edit - 0.2:
            return None
        if best_candidate['edit'] > max_allowed_edit:
            return None
        return best_candidate

    def _should_try_local_text_span_fallback(self, match_item):
        if match_item.get('gt_idx') == ['']:
            return False
        gt_category = match_item.get('gt_category_type')
        if gt_category not in ['title', 'text_block']:
            return False
        try:
            edit = float(match_item.get('edit', 1))
        except Exception:
            return False
        if edit <= 0.45:
            return False
        if gt_category == 'title':
            return True
        gt_text = str(match_item.get('gt', '') or '')
        norm_gt = str(match_item.get('norm_gt', '') or '')
        return self._text_contains_formula_hint(gt_text) or len(norm_gt) >= 80

    def _text_contains_formula_hint(self, text):
        text = str(text or '')
        if not text:
            return False
        if re.search('\\$[^$]+\\$|\\\\\\([^)]*\\\\\\)', text):
            return True
        if any((token in text for token in ['\\frac', '\\sum', '\\int', '\\sqrt', '\\left', '\\right'])):
            return True
        if ('^' in text or '_' in text) and ('$' in text or '\\' in text):
            return True
        return False

    def get_order_paired(self, order_match_s, img_name):
        matched = [(item['gt_position'], item['pred_position']) for item in order_match_s if item['gt_position'] != [''] and item['pred_position'] != '']
        gt_idx_all = [item['gt_position'] for item in order_match_s if item['gt_position'] != ['']]
        read_order_pred = [i[0] for i in sorted(matched, key=lambda x: x[1])]
        read_order_gt = sum(gt_idx_all, [])
        read_order_gt = [x for x in read_order_gt if x]
        gt = sorted(read_order_gt)
        pred = sum(read_order_pred, [])
        pred = [x for x in pred if x]
        if len(pred) > 0 or len(gt) > 0:
            edit = Levenshtein.distance(gt, pred) / max(len(pred), len(gt))
            return {'gt': gt, 'pred': pred, 'img_id': img_name, 'edit': edit}
        else:
            return {}
