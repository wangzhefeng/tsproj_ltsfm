import json
import sys
from pathlib import Path

ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data_provider.benchmark_dataset import build_series_windows, read_time_series_frame
from exp.exp_basic import Exp_Basic
from utils.log_util import logger
from utils.model_memory import model_memory_size
from utils.plot_results import predict_result_visual


class SimpleWindowEvalDataset(Dataset):

    def __init__(self, contexts, targets, start_indices, anchor_times):
        self.contexts = torch.from_numpy(contexts.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))
        self.start_indices = np.asarray(start_indices, dtype=np.int64)
        self.anchor_times = anchor_times or [""] * len(self.start_indices)

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, index):
        return {
            "inputs": self.contexts[index],
            "labels": self.targets[index],
            "start_index": int(self.start_indices[index]),
            "anchor_time": self.anchor_times[index],
        }


class SumEvalMetric:

    def __init__(self, name):
        self.name = name
        self.value = 0.0

    def push(self, preds, labels):
        self.value += self._calculate(preds, labels).item()

    def _calculate(self, preds, labels):
        raise NotImplementedError


class MSEMetric(SumEvalMetric):

    def _calculate(self, preds, labels):
        return torch.sum((preds - labels) ** 2)


class MAEMetric(SumEvalMetric):

    def _calculate(self, preds, labels):
        return torch.sum(torch.abs(preds - labels))


class Exp_Zero_Shot_Forecast_Simple(Exp_Basic):

    def __init__(self, args):
        logger.info(f"{40 * '-'}")
        logger.info("Initializing Simple Evaluation Experiment...")
        logger.info(f"{40 * '-'}")
        super().__init__(args)

    def _build_model(self):
        logger.info(f"Initializing model {self.args.model} for simple evaluation...")
        model = self.model_dict[self.args.model](self.args).float()
        if self.args.use_gpu and self.args.use_multi_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        model_memory_size(model, verbose=True)
        return model

    def _get_results_path(self, setting):
        results_path = Path(self.args.test_results).joinpath(setting)
        results_path.mkdir(parents=True, exist_ok=True)
        return results_path

    def _build_dataset(self):
        data_path = Path(self.args.root_path).joinpath(self.args.data_path)
        frame = read_time_series_frame(data_path, time_col=self.args.time)
        window_batch = build_series_windows(
            frame=frame,
            context_length=self.args.seq_len,
            prediction_length=self.args.pred_len,
            target_col=self.args.target,
            time_col=self.args.time,
            stride=self.args.testing_step,
            sample_limit=self.args.sample_limit,
        )
        window_batch.dataset_name = Path(self.args.data_path).stem
        dataset = SimpleWindowEvalDataset(
            contexts=window_batch.contexts,
            targets=window_batch.targets,
            start_indices=window_batch.start_indices,
            anchor_times=window_batch.time_index,
        )
        return window_batch, dataset

    @staticmethod
    def _to_model_inputs(inputs):
        if inputs.ndim == 2:
            return inputs.unsqueeze(-1)
        return inputs

    def _save_results(
        self,
        setting,
        path,
        metrics,
        preds,
        labels,
        start_indices,
        anchor_times,
        contexts,
    ):
        metrics_path = path.joinpath("metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as file:
            json.dump(metrics, file, ensure_ascii=False, indent=2)

        with open(path.joinpath("summary.txt"), "w", encoding="utf-8") as file:
            file.write(setting + "\n")
            for key, value in metrics.items():
                file.write(f"{key}: {value}\n")

        np.save(path.joinpath("predictions.npy"), preds)
        np.save(path.joinpath("labels.npy"), labels)
        np.save(path.joinpath("contexts.npy"), contexts)

        window_rows = pd.DataFrame(
            {
                "window_index": np.arange(len(start_indices)),
                "start_index": start_indices,
                "anchor_time": anchor_times,
            }
        )
        window_rows.to_csv(path.joinpath("windows.csv"), index=False, encoding="utf-8")

        sample_window = pd.DataFrame(
            {
                "step": np.arange(1, preds.shape[1] + 1),
                "pred": preds[0],
                "true": labels[0],
            }
        )
        sample_window.to_csv(path.joinpath("sample_prediction.csv"), index=False, encoding="utf-8")

        history = contexts[0]
        pred_plot = np.concatenate([history, preds[0]], axis=0)
        true_plot = np.concatenate([history, labels[0]], axis=0)
        predict_result_visual(pred_plot, true_plot, path=path, iters=None)

    def test(self, setting):
        window_batch, dataset = self._build_dataset()
        logger.info(f"{40 * '-'}")
        logger.info("Simple evaluation dataset ready")
        logger.info(f"dataset_name: {window_batch.dataset_name}")
        logger.info(f"num_windows: {len(dataset)}")
        logger.info(f"context_length: {self.args.seq_len}, prediction_length: {self.args.pred_len}")
        logger.info(f"{40 * '-'}")

        test_loader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=False,
        )
        results_path = self._get_results_path(setting)
        logger.info(f"Simple test results will be saved in: {results_path}")

        metric_list = [MSEMetric(name="mse"), MAEMetric(name="mae")]
        total_count = 0
        all_preds = []
        all_labels = []

        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                inputs = self._to_model_inputs(batch["inputs"].float().to(self.device))
                labels = batch["labels"].float().to(self.device)
                if labels.ndim == 2:
                    labels = labels.unsqueeze(-1)

                outputs = self.model(inputs, None, None, None)
                preds = outputs[:, -self.args.pred_len:, :]

                if preds.shape[-1] != labels.shape[-1]:
                    preds = preds[..., :labels.shape[-1]]

                for metric in metric_list:
                    metric.push(preds, labels)
                total_count += preds.numel()

                all_preds.append(preds.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())

        preds = np.concatenate(all_preds, axis=0).squeeze(-1)
        labels = np.concatenate(all_labels, axis=0).squeeze(-1)
        metrics = {
            "model": self.args.model,
            "data": str(Path(self.args.root_path).joinpath(self.args.data_path)),
            "target": window_batch.target_col,
            "context_length": self.args.seq_len,
            "prediction_length": self.args.pred_len,
            "batch_size": self.args.batch_size,
            "testing_step": self.args.testing_step,
            "num_windows": int(len(dataset)),
            "num_prediction_points": int(total_count),
        }
        for metric in metric_list:
            metrics[metric.name] = float(metric.value / total_count)

        self._save_results(
            setting=setting,
            path=results_path,
            metrics=metrics,
            preds=preds,
            labels=labels,
            start_indices=window_batch.start_indices,
            anchor_times=window_batch.time_index or [""] * len(window_batch.start_indices),
            contexts=window_batch.contexts,
        )
        logger.info(metrics)
        logger.info(f"{40 * '-'}")
        logger.info("Simple testing finished")
        logger.info(f"{40 * '-'}")
        return metrics
