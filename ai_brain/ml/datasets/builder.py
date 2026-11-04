"""
Dataset split builder and partitioning with grouping by source memories to prevent leakage.
"""

from collections import defaultdict
import random
from typing import List, Optional, Sequence, Tuple

from ai_brain.ml.datasets.schemas import DatasetExample, DatasetSplit
from ai_brain.ml.datasets.validator import DatasetValidator
from ai_brain.ml.exceptions import DatasetValidationError


class DatasetSplitter:
    """Partitions examples into train, validation, and test splits with strict group-level isolation."""

    @staticmethod
    def split(
        examples: Sequence[DatasetExample],
        *,
        name: str = "dataset",
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        seed: int = 42,
        group_by_memory: bool = True,
    ) -> DatasetSplit:
        if not examples:
            raise DatasetValidationError("Cannot split empty example collection")
        if val_ratio <= 0.0 or val_ratio >= 1.0:
            raise DatasetValidationError(f"Invalid val_ratio: {val_ratio}")
        if test_ratio < 0.0 or (val_ratio + test_ratio) >= 1.0:
            raise DatasetValidationError(f"Invalid test_ratio: {test_ratio}")

        feature_names = tuple(sorted(examples[0].features.keys()))

        rng = random.Random(seed)

        if not group_by_memory:
            shuffled = list(examples)
            rng.shuffle(shuffled)
            n_total = len(shuffled)
            n_val = max(1, int(n_total * val_ratio))
            n_test = int(n_total * test_ratio) if test_ratio > 0.0 else 0
            n_train = n_total - n_val - n_test

            train = shuffled[:n_train]
            val = shuffled[n_train:n_train + n_val]
            test = shuffled[n_train + n_val:]
        else:
            # Group by connected components of all source memory IDs so any examples
            # sharing ANY source memory ID are guaranteed to remain in the same split.
            groups = DatasetSplitter._find_connected_memory_groups(examples)
            rng.shuffle(groups)

            n_groups = len(groups)
            n_val_groups = max(1, int(n_groups * val_ratio))
            n_test_groups = int(n_groups * test_ratio) if test_ratio > 0.0 else 0
            n_train_groups = n_groups - n_val_groups - n_test_groups

            val_groups = groups[:n_val_groups]
            test_groups = groups[n_val_groups:n_val_groups + n_test_groups]
            train_groups = groups[n_val_groups + n_test_groups:]

            train, val, test = [], [], []
            for grp in train_groups:
                train.extend(grp)
            for grp in val_groups:
                val.extend(grp)
            for grp in test_groups:
                test.extend(grp)

            # Safety fallback if rounding left train empty
            if not train and val:
                train.append(val.pop())

        split_obj = DatasetSplit(
            name=name,
            train_examples=tuple(train),
            validation_examples=tuple(val),
            test_examples=tuple(test),
            feature_names=feature_names,
        )

        DatasetValidator.validate_split(split_obj)
        return split_obj

    @staticmethod
    def k_fold_split(
        examples: Sequence[DatasetExample],
        k: int = 5,
        *,
        name: str = "dataset",
        seed: int = 42,
        group_by_memory: bool = True,
    ) -> List[DatasetSplit]:
        """
        Creates k group-aware cross-validation splits.
        Guarantees that examples from the same source memory never cross between training and validation folds.
        """
        if not examples:
            raise DatasetValidationError("Cannot split empty example collection")
        if k < 2:
            raise DatasetValidationError(f"k must be at least 2, got {k}")

        feature_names = tuple(sorted(examples[0].features.keys()))
        rng = random.Random(seed)

        if not group_by_memory:
            shuffled = list(examples)
            rng.shuffle(shuffled)
            folds: List[List[DatasetExample]] = [[] for _ in range(k)]
            for i, ex in enumerate(shuffled):
                folds[i % k].append(ex)
        else:
            # Group by connected components of all source memory IDs
            groups = DatasetSplitter._find_connected_memory_groups(examples)
            rng.shuffle(groups)

            if len(groups) < k:
                raise DatasetValidationError(
                    f"Number of distinct connected memory groups ({len(groups)}) is less than k ({k})"
                )

            folds = [[] for _ in range(k)]
            for i, grp in enumerate(groups):
                folds[i % k].extend(grp)

        splits = []
        for i in range(k):
            val_fold = folds[i]
            train_fold = []
            for j in range(k):
                if j != i:
                    train_fold.extend(folds[j])

            split_obj = DatasetSplit(
                name=f"{name}_fold_{i+1}",
                train_examples=tuple(train_fold),
                validation_examples=tuple(val_fold),
                test_examples=(),
                feature_names=feature_names,
            )
            DatasetValidator.validate_split(split_obj)
            splits.append(split_obj)

        return splits

    @staticmethod
    def _find_connected_memory_groups(
        examples: Sequence[DatasetExample],
    ) -> List[List[DatasetExample]]:
        """
        Computes transitive connected components across all source memory IDs.
        Guarantees that if example A and example B share ANY source memory ID
        (directly or through a chain of shared IDs), they are placed in the same group.
        """
        mem_to_ex_indices = defaultdict(list)
        for idx, ex in enumerate(examples):
            mems = set(ex.provenance.source_memory_ids or ()) - {"", "none", None}
            if not mems:
                mem_to_ex_indices[f"__isolated_{ex.example_id}_{idx}"].append(idx)
            else:
                for m in mems:
                    mem_to_ex_indices[m].append(idx)

        parent = list(range(len(examples)))

        def find(i: int) -> int:
            path = []
            while parent[i] != i:
                path.append(i)
                i = parent[i]
            for node in path:
                parent[node] = i
            return i

        def union(i: int, j: int) -> None:
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j

        for mem, indices in mem_to_ex_indices.items():
            first = indices[0]
            for other in indices[1:]:
                union(first, other)

        groups_dict = defaultdict(list)
        for idx, ex in enumerate(examples):
            root = find(idx)
            groups_dict[root].append(ex)

        return list(groups_dict.values())


