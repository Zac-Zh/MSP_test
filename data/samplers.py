"""
Batch samplers for MSP detection training.

Provides balanced sampling strategies to ensure proper representation
of positive and negative samples in training batches.
"""

import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import Sampler, DataLoader
from utils.logging_utils import log_message


class CaseAwareBalancedBatchSampler(Sampler):
    """
    A sophisticated batch sampler that ensures:
    1. Slices from the same case (patient) are always in the same batch.
    2. Batches have a balanced composition of positive and negative cases.

    This version includes an explicit check for the 'patient_id' field.
    """

    def __init__(self, dataset, batch_size, pos_case_ratio=0.5):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.pos_case_ratio = pos_case_ratio

        log_file = self.dataset.config.get("log_file")

        # --- Final check: Ensure the required key exists in the dataset ---
        if not self.dataset.data_references:
            log_message("Warning: CaseAwareBalancedBatchSampler initialized with an empty dataset.", log_file)
            self.case_to_indices = {}
        else:
            # Use 'patient_id' as the grouping key. Assert it exists.
            key_to_group_by = 'patient_id'
            first_ref = self.dataset.data_references[0]
            if key_to_group_by not in first_ref:
                # If 'patient_id' is missing, fall back to 'case_id' but log a warning.
                key_to_group_by = 'case_id'
                log_message(f"Warning: '{key_to_group_by}' not found in data references. Falling back to 'case_id'.",
                            log_file)
                if key_to_group_by not in first_ref:
                    raise KeyError(
                        f"Neither 'patient_id' nor 'case_id' found in data references. Cannot group for sampler.")

            # Group slice indices by their case/patient_id
            self.case_to_indices = defaultdict(list)
            for i, ref in enumerate(self.dataset.data_references):
                self.case_to_indices[ref[key_to_group_by]].append(i)

        # Separate cases into positive (has at least one MSP slice) and negative
        self.positive_cases = []
        self.negative_cases = []
        for case_id, indices in self.case_to_indices.items():
            is_positive_case = any(self.dataset.data_references[i]['is_msp'] for i in indices)
            if is_positive_case:
                self.positive_cases.append(case_id)
            else:
                self.negative_cases.append(case_id)

        self.num_pos_cases_per_batch = int(self.batch_size * self.pos_case_ratio)
        self.num_neg_cases_per_batch = self.batch_size - self.num_pos_cases_per_batch

        num_possible_pos = len(
            self.positive_cases) // self.num_pos_cases_per_batch if self.num_pos_cases_per_batch > 0 else float('inf')
        num_possible_neg = len(
            self.negative_cases) // self.num_neg_cases_per_batch if self.num_neg_cases_per_batch > 0 else float('inf')

        self.num_batches = int(min(num_possible_pos, num_possible_neg))

        log_message(f"CaseAwareSampler: {len(self.positive_cases)} pos cases, {len(self.negative_cases)} neg cases. "
                    f"Batch composition: {self.num_pos_cases_per_batch} pos cases, {self.num_neg_cases_per_batch} neg cases. "
                    f"Creating {self.num_batches} batches.", log_file)

    def __iter__(self):
        if self.num_batches == 0:
            return iter([])

        np.random.shuffle(self.positive_cases)
        np.random.shuffle(self.negative_cases)

        pos_case_iter = iter(self.positive_cases)
        neg_case_iter = iter(self.negative_cases)

        for _ in range(self.num_batches):
            batch_indices = []

            try:
                for _ in range(self.num_pos_cases_per_batch):
                    batch_indices.extend(self.case_to_indices[next(pos_case_iter)])
                for _ in range(self.num_neg_cases_per_batch):
                    batch_indices.extend(self.case_to_indices[next(neg_case_iter)])
            except StopIteration:
                # This can happen if the number of cases is not perfectly divisible by the batch composition.
                # The self.num_batches calculation should prevent this, but this is a safeguard.
                break

            np.random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        return self.num_batches


class BalancedBatchSampler(torch.utils.data.Sampler):
    """
    ✅ 简化：只需要平衡正负样本，不再区分near/far
    """
    def __init__(self, dataset, batch_size: int, positive_ratio: float = 0.5):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size

        self.pos_indices = []
        self.neg_indices = []

        # 🔹 简化分类：只分正负
        for i, ref in enumerate(self.dataset.data_references):
            is_msp = ref.get("is_msp", False)
            if is_msp:
                self.pos_indices.append(i)
            else:
                self.neg_indices.append(i)

        self.pos_indices = np.array(self.pos_indices, dtype=np.int64)
        self.neg_indices = np.array(self.neg_indices, dtype=np.int64)

        log_file = self.dataset.config.get("log_file")

        # 计算每个batch的正负样本数量
        self.num_pos_per_batch = int(self.batch_size * positive_ratio)
        self.num_neg_per_batch = self.batch_size - self.num_pos_per_batch

        # 确定batch数量
        min_class_size = min(len(self.pos_indices), len(self.neg_indices))
        if min_class_size == 0:
            self.len = 0
        else:
            self.len = max(1, min_class_size // max(self.num_pos_per_batch, self.num_neg_per_batch))

        if log_file:
            log_message(f"BalancedBatchSampler: {len(self.pos_indices)} pos, "
                        f"{len(self.neg_indices)} neg indices. "
                        f"Batch composition: {self.num_pos_per_batch}:{self.num_neg_per_batch}",
                        log_file)

    def __iter__(self):
        for _ in range(self.len):
            batch = []

            # Sample positive
            if len(self.pos_indices) > 0 and self.num_pos_per_batch > 0:
                pos_selected = np.random.choice(self.pos_indices, size=self.num_pos_per_batch, replace=True)
                batch.extend(pos_selected.tolist())

            # Sample negative
            if len(self.neg_indices) > 0 and self.num_neg_per_batch > 0:
                neg_selected = np.random.choice(self.neg_indices, size=self.num_neg_per_batch, replace=True)
                batch.extend(neg_selected.tolist())

            if batch:
                np.random.shuffle(batch)
                yield batch

    def __len__(self):
        return self.len


def create_balanced_dataloader(dataset, config, is_train=True):
    """Creates a data loader that supports balanced sampling with soft labels."""
    # Import HeatmapDataset locally to avoid circular import
    from .datasets import HeatmapDataset

    if is_train and isinstance(dataset, HeatmapDataset):
        batch_sampler = BalancedBatchSampler(
            dataset,
            batch_size=config["BATCH_SIZE"]
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=config["NUM_WORKERS"],
            pin_memory=True if config["DEVICE"] == "cuda" else False,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=config["BATCH_SIZE"],
            shuffle=False,
            num_workers=config["NUM_WORKERS"],
            pin_memory=True if config["DEVICE"] == "cuda" else False,
            drop_last=False
        )
