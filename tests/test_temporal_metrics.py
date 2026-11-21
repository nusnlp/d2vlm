"""CPU regressions for the four D2VLM EVS/TAL evaluator copies.

Load the actual function definitions without importing caption scorers or
executing the conversion scripts' top-level CLI/data-loading code. Tensor
operations and temporal IoU use the real torch and nncore implementations.
"""

import ast
import itertools
import math
from pathlib import Path
import re
import unittest

import torch
from nncore.ops import temporal_iou


ROOT = Path(__file__).resolve().parents[1]
METRIC_FILES = (
    'all_benchmark_eval/etbench/new_compute_metrics.py',
    'all_benchmark_eval/etbench/new_compute_metircs_looser.py',
    'all_benchmark_eval/charades/convert_result_in_timechat_format.py',
    'all_benchmark_eval/youcook2/convert_result_in_timechat_format.py',
)
THRESHOLDS = (0.1, 0.3, 0.5, 0.7)
DELIMITER = '</evi>'


def load_metrics(path):
    names = {'tvg_format', 'evs_eval', 'tal_eval'}
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != names:
        raise AssertionError(f'Missing metric functions in {path}')
    namespace = dict(re=re, math=math, torch=torch, temporal_iou=temporal_iou)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def spans_text(spans):
    return ', '.join(f'{start}.0 - {end}.0' for start, end in spans)


def sample(gt, answer, evidence=None):
    evidence_text = '' if evidence is None else spans_text(evidence)
    return dict(tgt=gt, a=f'evidence {evidence_text} {DELIMITER} highlights: {spans_text(answer)}')


def original_samples(samples, evidence=False):
    """Give the shipped single-channel ETBench scorer only the chosen channel."""
    result = []
    for item in samples:
        evi, answer = item['a'].split(DELIMITER)
        text = evi if evidence else answer
        result.append(dict(tgt=item['tgt'], a=text))
    return result


def frame_f1(gt, pred):
    """Independent set-based oracle for the existing 1000-frame EVS metric."""
    def frames(spans):
        return {frame for start, end in spans for frame in range(max(0, round(start)), min(1000, round(end)))}

    gold, predicted = frames(gt), frames(pred)
    denominator = len(gold) + len(predicted)
    return 2 * len(gold & predicted) / denominator if denominator else 0.0


def segment_f1(gt, pred, threshold):
    """Independent scalar interval arithmetic; no torch/nncore IoU calls."""
    matches = []
    for start, end in gt:
        row = []
        for pred_start, pred_end in pred:
            intersection = max(0, min(end, pred_end) - max(start, pred_start))
            union = end - start + pred_end - pred_start - intersection
            row.append(intersection / union >= threshold)
        matches.append(row)
    recall = sum(any(row) for row in matches) / len(gt)
    precision = sum(any(column) for column in zip(*matches)) / len(pred)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


class MetricChecks:
    @classmethod
    def setUpClass(cls):
        cls.metrics = load_metrics(ROOT / cls.metric_path)
        cls.original = load_metrics(ROOT / 'all_benchmark_eval/etbench/compute_metrics_original_etchat.py')

    def check_evs(self, gt, answer, evidence, expected_answer, expected_evidence):
        inputs = [sample(gt, answer, evidence)]
        self.assertEqual(self.metrics['tvg_format'](inputs[0]['a'], DELIMITER), (answer, evidence))
        actual = self.metrics['evs_eval'](inputs, DELIMITER)
        self.assertEqual(actual['Total'], 1)
        self.assertEqual(actual['Failed'], 0)
        for use_evidence, spans, expected, key in (
            (False, answer, expected_answer, 'F1'),
            (True, evidence, expected_evidence, 'F1_evi'),
        ):
            self.assertAlmostEqual(frame_f1(gt, spans), expected)
            reference = self.original['evs_eval'](original_samples(inputs, use_evidence))
            self.assertEqual(reference['F1'], round(expected, 5))
            self.assertEqual(actual[key], reference['F1'])

    def check_tal(self, fixtures, expected_answer, expected_evidence):
        inputs = [sample(*fixture) for fixture in fixtures]
        actual = self.metrics['tal_eval'](inputs, DELIMITER)
        self.assertEqual(actual['Total'], len(fixtures))
        self.assertEqual(actual['Failed'], 0)
        for use_evidence, expected, prefix in (
            (False, expected_answer, 'F1'), (True, expected_evidence, 'F1_evi'),
        ):
            reference = self.original['tal_eval'](original_samples(inputs, use_evidence))
            for threshold, want in zip(THRESHOLDS, expected):
                independent = sum(segment_f1(gt, evi if use_evidence else ans, threshold)
                                  for gt, ans, evi in fixtures) / len(fixtures)
                self.assertAlmostEqual(independent, want)
                self.assertEqual(reference[f'F1@{threshold}'], round(want, 5))
                self.assertEqual(actual[f'{prefix}@{threshold}'], reference[f'F1@{threshold}'])
            self.assertEqual(actual[prefix], round(sum(expected) / len(expected), 5))
        return actual

    def test_evs_perfect_agreement(self):
        self.check_evs([[10, 20]], [[10, 20]], [[10, 20]], 1.0, 1.0)

    def test_evs_independent_masks(self):
        cases = (
            ([[50, 60]], [[50, 60]], [[10, 20]], 1.0, 0.0),
            ([[50, 60]], [[10, 20]], [[50, 60]], 0.0, 1.0),
            ([[0, 30]], [[0, 10]], [[0, 30]], 0.5, 1.0),
            ([[50, 60], [70, 75]], [[50, 60], [70, 75]], [[40, 80]], 1.0, 6 / 11),
            ([[10, 20]], [[15, 25]], [[30, 40]], 0.5, 0.0),
        )
        for case in cases:
            with self.subTest(case=case):
                self.check_evs(*case)

    def test_evs_multiple_examples(self):
        fixtures = [sample([[10, 20]], [[10, 20]], [[30, 40]]),
                    sample([[50, 60]], [[70, 80]], [[50, 60]])]
        for inputs in (fixtures, list(reversed(fixtures))):
            with self.subTest(inputs=inputs):
                actual = self.metrics['evs_eval'](inputs, DELIMITER)
                self.assertEqual(actual, dict(Total=2, Failed=0, F1=0.5, F1_evi=0.5))

    def test_tal_perfect_partial_and_nonmatching(self):
        cases = (
            (([[10, 20]], [[10, 20]], [[10, 20]]), [1.0] * 4, [1.0] * 4),
            (([[0, 10]], [[5, 15]], [[0, 10]]), [1.0, 1.0, 0.0, 0.0], [1.0] * 4),
            (([[10, 20], [30, 40]], [[10, 20]], [[30, 40]]), [2 / 3] * 4, [2 / 3] * 4),
            (([[10, 20]], [[30, 40]], [[50, 60]]), [0.0] * 4, [0.0] * 4),
        )
        for fixture, answer, evidence in cases:
            with self.subTest(fixture=fixture):
                self.check_tal([fixture], answer, evidence)

    def test_tal_evidence_miss_preserves_previous_scores(self):
        fixtures = [([[10, 20]], [[10, 20]], [[10, 20]]),
                    ([[30, 40]], [[30, 40]], [[30, 40]]),
                    ([[50, 60]], [[50, 60]], [[80, 90]])]
        self.check_tal(fixtures, [1.0] * 4, [2 / 3] * 4)

    def test_tal_evidence_scored_when_answer_misses(self):
        self.check_tal([([[70, 80]], [[0, 5]], [[70, 80]])], [0.0] * 4, [1.0] * 4)

    def test_tal_batch_is_order_independent(self):
        # Two hits, an evidence miss, then an answer miss: each channel has 3/4 hits.
        fixtures = [([[10, 20]], [[10, 20]], [[10, 20]]),
                    ([[30, 40]], [[30, 40]], [[30, 40]]),
                    ([[50, 60]], [[50, 60]], [[80, 90]]),
                    ([[70, 80]], [[0, 5]], [[70, 80]])]
        for permutation in itertools.permutations(fixtures):
            with self.subTest(permutation=permutation):
                self.check_tal(permutation, [0.75] * 4, [0.75] * 4)

    def test_unparseable_answer_denominators(self):
        valid = sample([[10, 20]], [[10, 20]], [[10, 20]])
        invalid = dict(tgt=[[10, 20]], a='no timestamps or delimiter')
        for name in ('evs_eval', 'tal_eval'):
            with self.subTest(metric=name):
                actual = self.metrics[name]([valid, invalid], DELIMITER)
                self.assertEqual(actual['Total'], 2)
                self.assertEqual(actual['Failed'], 1)
                # Keep the existing EVS valid-only / TAL all-samples denominators.
                expected = 1.0 if name == 'evs_eval' else 0.5
                self.assertEqual(actual['F1'], expected)
                self.assertEqual(actual['F1_evi'], expected)


class TestETBench(MetricChecks, unittest.TestCase):
    metric_path = METRIC_FILES[0]


class TestETBenchLooser(MetricChecks, unittest.TestCase):
    metric_path = METRIC_FILES[1]


class TestCharades(MetricChecks, unittest.TestCase):
    metric_path = METRIC_FILES[2]


class TestYouCook2(MetricChecks, unittest.TestCase):
    metric_path = METRIC_FILES[3]


class TestMetricConsistency(unittest.TestCase):
    def test_etbench_missing_evidence_fallback(self):
        # Only the ETBench copies implement this fallback; preserve that scope.
        inputs = [sample([[10, 20]], [[10, 20]])]
        for path in METRIC_FILES[:2]:
            metrics = load_metrics(ROOT / path)
            for name in ('evs_eval', 'tal_eval'):
                with self.subTest(path=path, metric=name):
                    actual = metrics[name](inputs, DELIMITER)
                    self.assertEqual(actual['F1'], 1.0)
                    self.assertEqual(actual['F1_evi'], 1.0)

    def test_all_copies_return_identical_metrics(self):
        inputs = [sample([[10, 20]], [[10, 20]], [[30, 40]]),
                  sample([[50, 60]], [[70, 80]], [[50, 60]])]
        outputs = []
        for path in METRIC_FILES:
            metrics = load_metrics(ROOT / path)
            outputs.append({name: metrics[name](inputs, DELIMITER) for name in ('evs_eval', 'tal_eval')})
        for path, output in zip(METRIC_FILES[1:], outputs[1:]):
            with self.subTest(path=path):
                self.assertEqual(output, outputs[0])


if __name__ == '__main__':
    unittest.main()
