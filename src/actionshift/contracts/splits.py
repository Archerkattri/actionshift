"""Immutable, hash-addressed ActionShift generalization splits."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np

from actionshift.contracts.sampler import contract_product, core_composition_product
from actionshift.contracts.types import ActionContract

SplitRule = Literal["seen", "unseen_value", "unseen_composition", "long_lag", "task_transfer"]
GENERATOR_VERSION = "split-v1"


def contract_hash(contract: ActionContract) -> str:
    return hashlib.sha256(contract.to_json().encode("utf-8")).hexdigest()


def composition_signature(contract: ActionContract) -> tuple[str, ...]:
    fields: list[str] = []
    if contract.permutation != tuple(range(len(contract.permutation))):
        fields.append("permutation")
    if contract.sign != (1,) * len(contract.sign):
        fields.append("sign")
    if contract.scale != (1.0,) * len(contract.scale):
        fields.append("scale")
    if contract.target != "delta":
        fields.append("target")
    if contract.frame != "base":
        fields.append("frame")
    if contract.lag != 0:
        fields.append("lag")
    if contract.gripper_inverted:
        fields.append("gripper")
    return tuple(fields)


@dataclass(frozen=True, slots=True)
class ContractRecord:
    contract: ActionContract
    sha256: str
    composition: tuple[str, ...]

    @classmethod
    def from_contract(cls, contract: ActionContract) -> ContractRecord:
        return cls(contract, contract_hash(contract), composition_signature(contract))

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": json.loads(self.contract.to_json()),
            "sha256": self.sha256,
            "composition": list(self.composition),
        }


@dataclass(frozen=True, slots=True)
class SplitManifest:
    schema_version: str
    generator_version: str
    seed: int
    rule: SplitRule
    train: tuple[ContractRecord, ...]
    test: tuple[ContractRecord, ...]
    manifest_sha256: str

    def _content(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generator_version": self.generator_version,
            "seed": self.seed,
            "rule": self.rule,
            "train": [record.to_dict() for record in self.train],
            "test": [record.to_dict() for record in self.test],
        }

    def to_json(self) -> str:
        return json.dumps(
            {**self._content(), "manifest_sha256": self.manifest_sha256},
            indent=2,
            sort_keys=True,
        ) + "\n"


def _manifest_hash(content: dict[str, Any]) -> str:
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _records(contracts: tuple[ActionContract, ...]) -> tuple[ContractRecord, ...]:
    return tuple(
        sorted(
            (ContractRecord.from_contract(item) for item in contracts),
            key=lambda item: item.sha256,
        )
    )


def _create_manifest(
    rule: SplitRule,
    seed: int,
    train: tuple[ActionContract, ...],
    test: tuple[ActionContract, ...],
) -> SplitManifest:
    provisional = SplitManifest(
        schema_version="1.0",
        generator_version=GENERATOR_VERSION,
        seed=seed,
        rule=rule,
        train=_records(train),
        test=_records(test),
        manifest_sha256="",
    )
    return SplitManifest(
        schema_version=provisional.schema_version,
        generator_version=provisional.generator_version,
        seed=provisional.seed,
        rule=provisional.rule,
        train=provisional.train,
        test=provisional.test,
        manifest_sha256=_manifest_hash(provisional._content()),
    )


def generate_split(rule: str, *, seed: int) -> SplitManifest:
    normalized = cast(SplitRule, rule)
    valid_rules = ("seen", "unseen_value", "unseen_composition", "long_lag", "task_transfer")
    if normalized not in valid_rules:
        raise ValueError(f"unknown split rule: {rule}")
    if normalized == "unseen_composition":
        contracts = core_composition_product()
        train = tuple(item for item in contracts if len(composition_signature(item)) <= 1)
        test = tuple(item for item in contracts if len(composition_signature(item)) >= 2)
        return _create_manifest(normalized, seed, train, test)
    contracts = contract_product()
    if normalized == "unseen_value":
        train = tuple(item for item in contracts if item.scale[0] != 0.5 and item.lag != 4)
        test = tuple(item for item in contracts if item.scale[0] == 0.5 or item.lag == 4)
    elif normalized == "long_lag":
        train = tuple(item for item in contracts if item.lag <= 1)
        test = tuple(item for item in contracts if item.lag >= 2)
    else:
        generator = np.random.default_rng(seed)
        ordering = generator.permutation(len(contracts))
        split = int(0.7 * len(contracts))
        train_indices = set(int(index) for index in ordering[:split])
        train = tuple(item for index, item in enumerate(contracts) if index in train_indices)
        test = tuple(item for index, item in enumerate(contracts) if index not in train_indices)
    return _create_manifest(normalized, seed, train, test)


def _field_values(records: tuple[ContractRecord, ...], field: str) -> set[str]:
    return {
        json.dumps(getattr(record.contract, field), sort_keys=True)
        for record in records
    }


def validate_manifest(manifest: SplitManifest) -> dict[str, Any]:
    train_hashes = {record.sha256 for record in manifest.train}
    test_hashes = {record.sha256 for record in manifest.test}
    train_compositions = {record.composition for record in manifest.train}
    test_compositions = {record.composition for record in manifest.test}
    all_records = (*manifest.train, *manifest.test)
    varying_fields = [
        field
        for field in ("permutation", "sign", "scale", "target", "frame", "lag", "gripper_inverted")
        if len(_field_values(all_records, field)) > 1
    ]
    support_valid = all(
        _field_values(manifest.train, field) == _field_values(all_records, field)
        for field in varying_fields
    )
    return {
        "schema_version": manifest.schema_version,
        "rule": manifest.rule,
        "train_count": len(manifest.train),
        "test_count": len(manifest.test),
        "overlap_count": len(train_hashes & test_hashes),
        "composition_overlap_count": len(train_compositions & test_compositions),
        "ordering_valid": (
            [item.sha256 for item in manifest.train] == sorted(train_hashes)
            and [item.sha256 for item in manifest.test] == sorted(test_hashes)
        ),
        "training_field_support_valid": support_valid,
        "record_hashes_valid": all(
            record.sha256 == contract_hash(record.contract) for record in all_records
        ),
        "manifest_hash_valid": manifest.manifest_sha256 == _manifest_hash(manifest._content()),
    }
