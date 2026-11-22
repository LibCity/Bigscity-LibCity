import unittest

import torch

from libcity.evaluator import TrafficAccidentEvaluator, eval_funcs


class AccidentTopKTests(unittest.TestCase):
    def check_metrics(self, predictions, labels, k, precision, recall):
        self.assertAlmostEqual(eval_funcs.Precision_torch(predictions, labels, k), precision)
        self.assertAlmostEqual(eval_funcs.Recall_torch(predictions, labels, k), recall)

    def test_all_positive_tied_grids(self):
        predictions = torch.full((1, 2, 2), 0.4)
        labels = torch.ones_like(predictions)
        for k in range(1, 5):
            with self.subTest(k=k):
                self.check_metrics(predictions, labels, k, 1, k / 4)

    def test_mixed_labels_with_ties_at_above_and_below_cutoff(self):
        labels = torch.tensor([[[1, 0, 2], [3, -1, 0]]])
        cases = (
            ([9, 5, 5, 5, 1, 0], 2),  # Ties span the cutoff.
            ([9, 9, 5, 4, 1, 0], 3),  # Ties are inside the cutoff.
            ([9, 8, 5, 5, 5, 0], 2),  # Ties are below the cutoff.
        )
        for scores, k in cases:
            with self.subTest(scores=scores, k=k):
                predictions = torch.tensor(scores, dtype=torch.float32).reshape(1, 2, 3)
                indices = torch.topk(predictions.reshape(-1), k).indices.tolist()
                self.assertEqual(len(indices), k)
                self.assertEqual(len(set(indices)), k)
                relevance = [value > 0 for value in labels.reshape(-1).tolist()]
                hits = sum(relevance[index] for index in indices)
                self.check_metrics(predictions, labels, k, hits / k, hits / sum(relevance))

    def test_untied_ranking(self):
        predictions = torch.tensor([[[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]])
        cases = (
            ([1, 0, 0, 0, 0, 0], 1, 1, 1),        # Relevant at rank 1.
            ([0, 1, 0, 0, 0, 0], 2, 0.5, 1),      # Relevant at rank K.
            ([0, 0, 1, 0, 0, 0], 2, 0, 0),        # Relevant just outside K.
            ([1, 2, 1, 0, 0, 0], 2, 1, 2 / 3),    # Several hits.
            ([0, -1, 0, 0, 0, 1], 2, 0, 0),       # No hits in top K.
            ([1, 1, 1, 1, 1, 1], 3, 1, 0.5),      # All grids relevant.
        )
        for values, k, precision, recall in cases:
            with self.subTest(values=values, k=k):
                labels = torch.tensor(values).reshape(1, 2, 3)
                self.check_metrics(predictions, labels, k, precision, recall)

    def test_batch_averaging_and_zero_relevance_skip(self):
        predictions = torch.tensor([
            list(range(9, 0, -1)),
            list(range(-5, 4)),
            list(range(9)),
        ], dtype=torch.float32).reshape(3, 3, 3).transpose(1, 2)
        labels = torch.tensor([
            [1, 2, 0, 0, -1, 0, 1, 1, 0],
            [-1, 1, 2, 0, 0, -2, 0, 1, 0],
            [0, 0, 0, 0, 0, -1, 0, -2, 0],
        ]).reshape(3, 3, 3).transpose(1, 2)
        self.check_metrics(predictions, labels, 2, (1 + 0.5 + 0) / 3, (0.5 + 1 / 3) / 2)

    def test_single_mode_evaluator_collects_exact_topk_metrics(self):
        evaluator = TrafficAccidentEvaluator({
            "metrics": ["Precision", "Recall", "F1-Score"],
            "evaluator_mode": "single",
            "topk": 2,
        })
        predictions = torch.tensor([[[[0.4, 0.4], [0.4, 0.4]], [[4.0, 3.0], [2.0, 1.0]]]])
        labels = torch.tensor([[[[1, 1], [1, 1]], [[1, 0], [1, 0]]]])
        evaluator.collect({"y_pred": predictions.unsqueeze(-1), "y_true": labels.unsqueeze(-1)})
        expected = {
            "Precision@1": 1,
            "Recall@1": 0.5,
            "F1-Score@1": 2 / 3,
            "Precision@2": 0.5,
            "Recall@2": 0.5,
            "F1-Score@2": 0.5,
        }
        self.assertEqual(evaluator.len_timeslots, 2)
        self.assertEqual(evaluator.intermediate_result.keys(), expected.keys())
        for metric, value in expected.items():
            with self.subTest(metric=metric):
                self.assertEqual(len(evaluator.intermediate_result[metric]), 1)
                self.assertAlmostEqual(evaluator.intermediate_result[metric][0], value)

    def test_map_and_pcc_untied_controls(self):
        predictions = torch.tensor([[[4.0, 3.0], [2.0, 1.0]]])
        labels = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        self.assertAlmostEqual(eval_funcs.MAP_torch(predictions, labels, 2), 0.5)
        self.assertAlmostEqual(eval_funcs.PCC_torch(predictions, labels, 2), 1, delta=1e-6)

    def test_out_of_range_cutoffs_keep_legacy_behavior(self):
        predictions = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        labels = torch.tensor([[1, 0, 1, -1]])
        for metric in (eval_funcs.Precision_torch, eval_funcs.Recall_torch):
            with self.assertRaises(IndexError):
                metric(predictions, labels, 5)
        with self.assertRaises(ZeroDivisionError):
            eval_funcs.Precision_torch(predictions, labels, 0)
        self.assertEqual(eval_funcs.Recall_torch(predictions, labels, 0), 1)
        self.check_metrics(predictions, labels, -1, -2, 1)

    def test_nan_does_not_count_as_a_hit(self):
        predictions = torch.tensor([[float("nan"), 3.0, 2.0, 1.0]])
        labels = torch.tensor([[1, 1, 1, -1]])
        self.check_metrics(predictions, labels, 1, 0, 0)
        self.check_metrics(predictions, labels, 2, 0.5, 1 / 3)


if __name__ == "__main__":
    unittest.main()
