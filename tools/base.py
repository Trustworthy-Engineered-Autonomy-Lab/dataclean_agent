import inspect
import json
import re


def _schema_errors(schema, path, root=False):
    """Check the portable function-schema shape, not experimental policy.

    This is deliberately not a strict-mode schema converter: optional fields
    and free-form objects remain optional/free-form. Never silently strip a
    constraint from a malformed tool definition.
    """
    errors = []
    if not isinstance(schema, dict):
        return [f"{path}: schema must be an object"]
    if root:
        if schema.get("type") != "object":
            errors.append(f"{path}: function parameters must have type=object")
        for key in sorted(set(schema) & {"oneOf", "anyOf", "allOf", "enum", "const", "not"}):
            errors.append(f"{path}.{key}: not supported at function-parameter root")
    types = schema.get("type")
    if types is not None:
        types = types if isinstance(types, list) else [types]
        if not types or any(t not in ("object", "array", "string", "number", "integer", "boolean", "null") for t in types):
            errors.append(f"{path}.type: invalid JSON Schema type")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        errors.append(f"{path}.properties: must be an object")
        properties = {}
    for name, child in properties.items():
        errors.extend(_schema_errors(child, f"{path}.properties.{name}"))
    required = schema.get("required", [])
    if not isinstance(required, list) or any(not isinstance(k, str) for k in required):
        errors.append(f"{path}.required: must be a list of property names")
    elif len(required) != len(set(required)) or set(required) - set(properties):
        errors.append(f"{path}.required: duplicate or undefined property names")
    if types and "array" in types and "items" not in schema:
        errors.append(f"{path}.items: array schema must declare its items")
    if "items" in schema:
        errors.extend(_schema_errors(schema["items"], f"{path}.items"))
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        errors.append(f"{path}.enum: must be a non-empty list")
    additional = schema.get("additionalProperties", True)
    if isinstance(additional, dict):
        errors.extend(_schema_errors(additional, f"{path}.additionalProperties"))
    elif not isinstance(additional, bool):
        errors.append(f"{path}.additionalProperties: must be a boolean or schema")
    for key in ("oneOf", "anyOf", "allOf"):
        if key not in schema:
            continue
        branches = schema[key]
        if not isinstance(branches, list) or not branches:
            errors.append(f"{path}.{key}: must be a non-empty list of schemas")
        else:
            for index, child in enumerate(branches):
                errors.extend(_schema_errors(child, f"{path}.{key}[{index}]"))
    return errors


class Tool:
    name: str = ""
    description: str = ""
    parameters: dict = {}
    _registry: dict = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "name") and cls.name:
            Tool._registry[cls.name] = cls

    @classmethod
    def get(cls, name: str):
        if name not in cls._registry:
            raise KeyError(f"Tool '{name}' is not registered in Tool._registry.")
        return cls._registry[name]()

    @classmethod
    def schema_errors(cls):
        """Read-only preflight shared by requests, doctor and the health API."""
        errors = []
        for tool_cls in cls._registry.values():
            if not tool_cls.name or not getattr(tool_cls, "agent_exposed", True):
                continue
            name = tool_cls.name
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
                errors.append(f"{name!r}: invalid function name")
            try:
                json.dumps(tool_cls.parameters, allow_nan=False)
            except (ValueError, TypeError) as exc:
                errors.append(f"{name}.parameters: not valid JSON ({exc})")
                continue
            errors.extend(_schema_errors(tool_cls.parameters, f"{name}.parameters", root=True))
            properties = tool_cls.parameters.get("properties", {}) if isinstance(tool_cls.parameters, dict) else {}
            signature = inspect.signature(tool_cls.run)
            if isinstance(properties, dict):
                for field in properties:
                    if field not in signature.parameters:
                        errors.append(f"{name}.parameters.{field}: no matching run() argument")
                for field, parameter in signature.parameters.items():
                    if (field != "self" and parameter.default is inspect.Parameter.empty
                            and parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
                            and field not in properties):
                        errors.append(f"{name}.run({field}): required argument missing from schema")
        return errors

    @classmethod
    def all_schemas(cls):
        errors = cls.schema_errors()
        if errors:
            raise ValueError("Invalid tool definitions: " + "; ".join(errors))
        return [
            {
                "type": "function",
                "function": {
                    "name": tool_cls.name,
                    "description": tool_cls.description,
                    "parameters": tool_cls.parameters
                }
            }
            for tool_cls in cls._registry.values()
            if tool_cls.name and getattr(tool_cls, "agent_exposed", True)
        ]

    def run(self, **kwargs):
        raise NotImplementedError("Tool run method must be implemented.")
