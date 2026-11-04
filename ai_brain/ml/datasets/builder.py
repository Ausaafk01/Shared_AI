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
            # Group by root/source memory ID so examples concerning the same memory are kept in the same split
            groups = defaultdict(list)
            for ex in examples:
                if ex.provenance.source_memory_ids:
                    key = ex.provenance.source_memory_ids[0]
                else:
                    key = ex.example_id
                groups[key].append(ex)

            group_keys = list(groups.keys())
            rng.shuffle(group_keys)

            n_groups = len(group_keys)
            n_val_groups = max(1, int(n_groups * val_ratio))
            n_test_groups = int(n_groups * test_ratio) if test_ratio > 0.0 else 0
            n_train_groups = n_groups - n_val_groups - n_test_groups

            val_keys = set(group_keys[:n_val_groups])
            test_keys = set(group_keys[n_val_groups:n_val_groups + n_test_groups])

            train, val, test = [], [], []
            for k, grp_examples in groups.items():
                if k in val_keys:
                    val.extend(grp_examples)
                elif k in test_keys:
                    test.extend(grp_examples)
                else:
                    train.extend(grp_examples)

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
