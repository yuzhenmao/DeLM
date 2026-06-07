# Adapted from FoundationAgents/AOrchestra:
# https://github.com/FoundationAgents/AOrchestra
#
# Copyright notice and license of the original project are retained
# in accordance with the Apache License, Version 2.0.
# This file includes modifications for the current project.

"""Data loader for SWE-bench Verified dataset."""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import json

from base.engine.logs import logger


@dataclass
class SWEBenchInstance:
    """Represents a single SWE-bench instance."""
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: str
    created_at: str
    patch: str  # Gold patch for reference (not shown to agent)
    test_patch: str  # Test patch to apply
    version: str
    environment_setup_commit: Optional[str] = None
    FAIL_TO_PASS: Optional[List[str]] = None
    PASS_TO_PASS: Optional[List[str]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SWEBenchInstance":
        """Create instance from dataset dict."""
        # Handle FAIL_TO_PASS and PASS_TO_PASS which may be JSON strings
        fail_to_pass = data.get("FAIL_TO_PASS")
        if isinstance(fail_to_pass, str):
            try:
                fail_to_pass = json.loads(fail_to_pass)
            except json.JSONDecodeError:
                fail_to_pass = [fail_to_pass] if fail_to_pass else []
        
        pass_to_pass = data.get("PASS_TO_PASS")
        if isinstance(pass_to_pass, str):
            try:
                pass_to_pass = json.loads(pass_to_pass)
            except json.JSONDecodeError:
                pass_to_pass = [pass_to_pass] if pass_to_pass else []
        
        return cls(
            instance_id=data["instance_id"],
            repo=data["repo"],
            base_commit=data["base_commit"],
            problem_statement=data["problem_statement"],
            hints_text=data.get("hints_text", ""),
            created_at=data.get("created_at", ""),
            patch=data.get("patch", ""),
            test_patch=data.get("test_patch", ""),
            version=data.get("version", ""),
            environment_setup_commit=data.get("environment_setup_commit"),
            FAIL_TO_PASS=fail_to_pass,
            PASS_TO_PASS=pass_to_pass,
        )


class SWEBenchDataLoader:
    """Loader for SWE-bench Verified dataset from Hugging Face."""
    
    def __init__(
        self,
        dataset_name: str = "princeton-nlp/SWE-bench_Verified",
        split: str = "test",
        cache_dir: Optional[str] = None,
        subset_seed: Optional[int] = None,
        subset_sizes: Optional[Dict[str, int]] = None,
        subset_role: Optional[str] = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.cache_dir = cache_dir
        self.subset_seed = subset_seed
        self.subset_sizes = subset_sizes
        self.subset_role = subset_role.lower() if subset_role else None
        self._dataset = None
        self._instances: Optional[List[SWEBenchInstance]] = None
    
    def _load_dataset(self):
        """Load dataset from Hugging Face."""
        if self._dataset is not None:
            return
        
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "Please install the 'datasets' package: pip install datasets"
            )
        
        logger.info(f"Loading dataset: {self.dataset_name} (split: {self.split})")
        self._dataset = load_dataset(
            self.dataset_name,
            split=self.split,
            cache_dir=self.cache_dir,
        )
        logger.info(f"Loaded {len(self._dataset)} instances")
    
    def load_instances(self) -> List[SWEBenchInstance]:
        """Load all instances from the dataset."""
        if self._instances is not None:
            return self._instances
        
        self._load_dataset()
        dataset = self._dataset

        if self.subset_sizes and self.subset_role:
            role = self.subset_role
            sizes = {k.lower(): v for k, v in self.subset_sizes.items() if v}
            ordered_sizes = list(sizes.items())
            known_roles = set(sizes.keys())
            combined = role in {"all", "combined"}
            if role not in known_roles and not combined:
                logger.warning(f"Requested subset '{role}' not found in sizes {list(known_roles)}. Using full split.")
            else:
                seed = self.subset_seed or 0
                shuffled = dataset.shuffle(seed=seed)
                total_needed = sum(count for _, count in ordered_sizes)
                available = len(shuffled)
                if available < total_needed:
                    logger.warning(
                        f"Requested {total_needed} instances across subsets but only {available} available; truncating."
                    )
                cursor = 0
                ranges = {}
                for name, count in ordered_sizes:
                    start = cursor
                    end = min(cursor + count, available)
                    ranges[name] = (start, end)
                    cursor += count
                if combined:
                    start, end = 0, min(total_needed, available)
                    label = "+".join(name for name, _ in ordered_sizes)
                else:
                    start, end = ranges.get(role, (0, available))
                    label = role
                if start >= available or start >= end:
                    logger.warning(f"Subset '{role}' start {start} beyond dataset size {available}; using empty subset.")
                    dataset = shuffled.select([])
                else:
                    dataset = shuffled.select(range(start, end))
                    logger.info(
                        f"Using subset '{label}' with {end - start} instances "
                        f"(seed={seed}, sizes={sizes}, span={start}:{end})"
                    )

        self._instances = [
            SWEBenchInstance.from_dict(item)
            for item in dataset
        ]
        return self._instances
    
    def get_instance(self, instance_id: str) -> Optional[SWEBenchInstance]:
        """Get a specific instance by ID."""
        instances = self.load_instances()
        for inst in instances:
            if inst.instance_id == instance_id:
                return inst
        return None
    
    def list_instance_ids(self) -> List[str]:
        """List all instance IDs."""
        instances = self.load_instances()
        return [inst.instance_id for inst in instances]
    
    def __len__(self) -> int:
        """Return number of instances."""
        self._load_dataset()
        return len(self._dataset)
    
    def __iter__(self):
        """Iterate over instances."""
        return iter(self.load_instances())
