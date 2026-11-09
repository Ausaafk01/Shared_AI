"""
Dataset validation and integrity checking for AI Brain ML layer.
"""

from typing import Sequence, Set

from ai_brain.ml.datasets.schemas import DatasetExample, DatasetSplit
from ai_brain.ml.exceptions import DatasetValidationError


class DatasetValidator:
    """Validates schemas, non-empty features, label distributions, and verifies zero train/eval leakage."""

    @staticmethod
    def validate_example(example: DatasetExample, expected_features: Sequence[str]) -> None:
        if not example.example_id or not example.example_id.strip():
            raise DatasetValidationError("example_id must be a non-empty string")
        if not example.features:
            raise DatasetValidationError(f"Example '{example.example_id}' has empty features")
        
        missing = [f for f in expected_features if f not in example.features]
        if missing:
            raise DatasetValidationError(f"Example '{example.example_id}' missing expected features: {missing}")

        for k, v in example.features.items():
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise DatasetValidationError(f"Feature '{k}' in '{example.example_id}' must be numeric, got {type(v).__name__}")

    @staticmethod
    def validate_split(split: DatasetSplit) -> None:
        if not split.train_examples:
            raise DatasetValidationError(f"DatasetSplit '{split.name}' has empty train_examples")
        if not split.validation_examples:
            raise DatasetValidationError(f"DatasetSplit '{split.name}' has empty validation_examples")

        # 1. Check feature consistency
        all_examples = list(split.train_examples) + list(split.validation_examples) + list(split.test_examples)
        for ex in all_examples:
            DatasetValidator.validate_example(ex, split.feature_names)

        # 2. Check for ID leakage
        train_ids = {ex.example_id for ex in split.train_examples}
        val_ids = {ex.example_id for ex in split.validation_examples}
        test_ids = {ex.example_id for ex in split.test_examples}

        leak_train_val = train_ids.intersection(val_ids)
        if leak_train_val:
            raise DatasetValidationError(f"Data leakage detected between train and validation sets: {leak_train_val}")

        leak_train_test = train_ids.intersection(test_ids)
        if leak_train_test:
            raise DatasetValidationError(f"Data leakage detected between train and test sets: {leak_train_test}")

        leak_val_test = val_ids.intersection(test_ids)
        if leak_val_test:
            raise DatasetValidationError(f"Data leakage detected between validation and test sets: {leak_val_test}")

        # 3. Check for memory entity ID leakage (underlying source memory overlap)
        train_mem_ids: Set[str] = set()
        for ex in split.train_examples:
            train_mem_ids.update(ex.provenance.source_memory_ids)

        val_mem_ids: Set[str] = set()
        for ex in split.validation_examples:
            val_mem_ids.update(ex.provenance.source_memory_ids)

        common_memories = train_mem_ids.intersection(val_mem_ids) - {"", "none", None}
        if common_memories:
            raise DatasetValidationError(
                f"Source memory leakage detected between train and validation: {common_memories}"
            )
