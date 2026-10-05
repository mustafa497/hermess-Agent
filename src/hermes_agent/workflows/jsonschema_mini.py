"""A small JSON Schema validator for workflow step outputs.

Ollama's `format` parameter already constrains generation to the schema, so
this is a verification pass, not a parser. It covers the subset workflow
authors actually use:

    type (incl. lists), properties, required, items, enum, additionalProperties,
    minimum / maximum, minLength / maxLength, minItems / maxItems

Anything else in a schema is passed to Ollama untouched and simply not checked
here. If you need full Draft 2020-12 semantics, install `jsonschema` and swap
`validate` out -- the call site is one function.
"""

from __future__ import annotations

from typing import Any

_TYPE_CHECKS: dict[str, Any] = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def validate(instance: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Return a list of human-readable problems; empty means valid."""
    if not isinstance(schema, dict) or not schema:
        return []

    errors: list[str] = []

    for combinator in ("anyOf", "oneOf"):
        options = schema.get(combinator)
        if isinstance(options, list) and options:
            variants = [o for o in options if isinstance(o, dict)]
            if not any(not validate(instance, v, path) for v in variants):
                errors.append(f"{path}: does not match any schema in {combinator}")
            return errors

    expected = schema.get("type")
    if expected:
        candidates = expected if isinstance(expected, list) else [expected]
        checks = [_TYPE_CHECKS[t] for t in candidates if t in _TYPE_CHECKS]
        if checks and not any(check(instance) for check in checks):
            errors.append(
                f"{path}: expected type {'|'.join(candidates)}, got {_name(instance)}"
            )
            return errors  # further checks would be noise

    enum = schema.get("enum")
    if isinstance(enum, list) and instance not in enum:
        errors.append(f"{path}: {instance!r} is not one of {enum}")

    if isinstance(instance, dict):
        errors.extend(_check_object(instance, schema, path))
    elif isinstance(instance, list):
        errors.extend(_check_array(instance, schema, path))
    elif isinstance(instance, str):
        errors.extend(_check_string(instance, schema, path))
    elif isinstance(instance, (int, float)) and not isinstance(instance, bool):
        errors.extend(_check_number(instance, schema, path))

    return errors


def _name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _check_object(instance: dict[str, Any], schema: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    properties = schema.get("properties") or {}

    for key in schema.get("required") or []:
        if key not in instance:
            errors.append(f"{path}.{key}: required property is missing")

    for key, value in instance.items():
        subschema = properties.get(key)
        if isinstance(subschema, dict):
            errors.extend(validate(value, subschema, f"{path}.{key}"))
        elif schema.get("additionalProperties") is False and properties:
            errors.append(f"{path}.{key}: additional property is not allowed")
    return errors


def _check_array(instance: list[Any], schema: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    items = schema.get("items")
    if isinstance(items, dict):
        for i, value in enumerate(instance):
            errors.extend(validate(value, items, f"{path}[{i}]"))
    min_items, max_items = schema.get("minItems"), schema.get("maxItems")
    if isinstance(min_items, int) and len(instance) < min_items:
        errors.append(f"{path}: expected at least {min_items} items, got {len(instance)}")
    if isinstance(max_items, int) and len(instance) > max_items:
        errors.append(f"{path}: expected at most {max_items} items, got {len(instance)}")
    return errors


def _check_string(instance: str, schema: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    min_len, max_len = schema.get("minLength"), schema.get("maxLength")
    if isinstance(min_len, int) and len(instance) < min_len:
        errors.append(f"{path}: shorter than minLength {min_len}")
    if isinstance(max_len, int) and len(instance) > max_len:
        errors.append(f"{path}: longer than maxLength {max_len}")
    return errors


def _check_number(instance: float, schema: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    minimum, maximum = schema.get("minimum"), schema.get("maximum")
    if isinstance(minimum, (int, float)) and instance < minimum:
        errors.append(f"{path}: {instance} is below minimum {minimum}")
    if isinstance(maximum, (int, float)) and instance > maximum:
        errors.append(f"{path}: {instance} is above maximum {maximum}")
    return errors
