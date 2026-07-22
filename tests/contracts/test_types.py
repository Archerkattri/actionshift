from __future__ import annotations

import json

import pytest

from actionshift.contracts.types import ActionContract


def valid_contract(**overrides: object) -> ActionContract:
    fields: dict[str, object] = {
        "permutation": (1, 0),
        "sign": (1, -1),
        "scale": (1.0, 0.01),
        "target": "delta",
        "frame": "base",
        "lag": 2,
        "gripper_inverted": False,
    }
    fields.update(overrides)
    return ActionContract(**fields)  # type: ignore[arg-type]


def test_contract_is_hashable_and_immutable() -> None:
    contract = valid_contract()

    assert hash(contract)
    with pytest.raises((AttributeError, TypeError)):
        contract.lag = 3  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("permutation", (0, 0), "permutation must be a bijection"),
        ("sign", (1, 0), "sign values must be -1 or 1"),
        ("scale", (1.0, 0.0), "scale values must be finite and positive"),
        ("scale", (1.0, float("inf")), "scale values must be finite and positive"),
        ("lag", -1, "lag must be nonnegative"),
        ("target", "velocity", "target must be 'delta' or 'absolute'"),
        ("frame", "world", "frame must be 'base' or 'tool'"),
    ],
)
def test_contract_rejects_invalid_fields(field: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        valid_contract(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sign", (1,)),
        ("scale", (1.0,)),
    ],
)
def test_contract_rejects_dimension_mismatch(field: str, value: object) -> None:
    with pytest.raises(ValueError, match="contract component dimensions must match"):
        valid_contract(**{field: value})


def test_contract_rejects_empty_pose_dimension() -> None:
    with pytest.raises(ValueError, match="contract must have at least one pose component"):
        valid_contract(permutation=(), sign=(), scale=())


def test_contract_round_trips_through_stable_json() -> None:
    contract = valid_contract()

    encoded = contract.to_json()
    decoded = ActionContract.from_json(encoded)

    assert decoded == contract
    assert json.loads(encoded) == {
        "frame": "base",
        "gripper_inverted": False,
        "lag": 2,
        "permutation": [1, 0],
        "scale": [1.0, 0.01],
        "sign": [1, -1],
        "target": "delta",
    }
    assert encoded == contract.to_json()


def test_from_json_rejects_unknown_fields() -> None:
    payload = json.loads(valid_contract().to_json())
    payload["oracle_contract_id"] = 7

    with pytest.raises(ValueError, match="unknown contract fields: oracle_contract_id"):
        ActionContract.from_json(json.dumps(payload))


def test_contract_rejects_non_boolean_gripper_flag() -> None:
    with pytest.raises(ValueError, match="gripper_inverted must be boolean"):
        valid_contract(gripper_inverted=1)


def test_from_json_rejects_fractional_lag_instead_of_truncating_it() -> None:
    payload = json.loads(valid_contract().to_json())
    payload["lag"] = 1.5

    with pytest.raises(ValueError, match="lag must be an integer"):
        ActionContract.from_json(json.dumps(payload))
