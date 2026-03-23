import json
from typing import Any, Dict, Tuple, Optional

# Define schema for each action
ACTION_SCHEMA = {
    "ExfiltrateData": {
        "required": ["target_host", "source_host", "data"],
        "schema": {
            "target_host": str,
            "source_host": str,
            "data": {  # <-- Make this a dictionary schema
                "required": ["owner", "id"],  # both fields required
                "schema": {
                    "owner": str,
                    "id": str
                }
            }
        }
    },
    "FindServices": {
        "required": ["target_host", "source_host"],
        "schema": {
            "target_host": str,
            "source_host": str
        }
    },
    "FindData": {
        "required": ["target_host", "source_host"],
        "schema": {
            "target_host": str,
            "source_host": str
        }
    },
    "ExploitService": {
        "required": ["target_host", "target_service", "source_host"],
        "schema": {
            "target_host": str,
            "target_service": str,
            "source_host": str
        }
    },
    "ScanServices": {
        "required": ["target_host", "source_host"],
        "schema": {
            "target_host": str,
            "source_host": str
        }
    },
    "ScanNetwork": {
        "required": ["target_network", "source_host"],
        "schema": {
            "target_network": str,
            "source_host": str
        }
    }
}


def _normalize_exfiltrate_data(parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> None:
    """
    Normalize parameters["data"] for ExfiltrateData and optionally fill missing fields.

    - Accepts data as a JSON string, comma-separated string "owner, id", or dict
    - If id is missing and context provides a unique candidate id for the given owner and source_host,
      fills it into the structure.
    """
    if not isinstance(parameters, dict):
        return

    data_field = parameters.get("data")
    # If 'data' is a JSON string; try to decode
    if isinstance(data_field, str):
        s = data_field.strip()
        parsed = None
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                parsed = json.loads(s)
            except Exception:
                parsed = None
        if parsed is not None:
            data_field = parsed
        else:
            # Accept comma-separated owner, id
            if "," in s:
                parts = [p.strip() for p in s.split(",")]
                if len(parts) >= 2:
                    data_field = {"owner": parts[0], "id": parts[1]}
                elif len(parts) == 1:
                    data_field = {"owner": parts[0]}

    # Ensure dict structure going forward
    if not isinstance(data_field, dict):
        return

    # Coerce simple types to strings
    if "owner" in data_field and not isinstance(data_field["owner"], str):
        try:
            data_field["owner"] = str(data_field["owner"])
        except Exception:
            pass
    if "id" in data_field and not isinstance(data_field["id"], str):
        try:
            data_field["id"] = str(data_field["id"])
        except Exception:
            pass

    # Fill missing id when possible using provided context
    if ("id" not in data_field or not isinstance(data_field.get("id"), str)) and context:
        try:
            owner = data_field.get("owner")
            src_host = parameters.get("source_host")
            if isinstance(owner, str) and isinstance(src_host, str):
                kd_map = context.get("known_data_map", {})
                candidates = []
                for d in kd_map.get(src_host, []):
                    if isinstance(d, dict) and d.get("owner") == owner:
                        cid = d.get("id")
                        if isinstance(cid, str):
                            candidates.append(cid)
                # Only auto-fill when unambiguous
                if len(candidates) == 1:
                    data_field["id"] = candidates[0]
        except Exception:
            pass

    parameters["data"] = data_field


def validate_schema(data: dict, schema: dict) -> Tuple[bool, str | bool]:
    """
    Recursively validate that data matches the schema.
    """
    for key, expected_type in schema.items():
        if key not in data:
            return False, f"Error: Missing required key '{key}'"

        value = data[key]

        if isinstance(expected_type, dict):
            # Handle nested schema with "schema" and "required"
            if "schema" in expected_type:
                if not isinstance(value, dict):
                    return False, f"Error: Field '{key}' must be a dictionary"

                # Validate required fields
                required = expected_type.get("required", [])
                for req_key in required:
                    if req_key not in value:
                        return False, f"Error: Missing required key '{req_key}' in '{key}'"

                # Recursively validate inner schema
                inner_result = validate_schema(value, expected_type["schema"])
                if not inner_result[0]:
                    return inner_result

            else:
                # Regular nested dictionary check
                if not isinstance(value, dict):
                    return False, f"Error: Field '{key}' must be a dictionary"
                inner_result = validate_schema(value, expected_type)
                if not inner_result[0]:
                    return inner_result

        elif isinstance(expected_type, type):
            if not isinstance(value, expected_type):
                return False, f"Error: Field '{key}' must be of type {expected_type.__name__}"
        else:
            return False, f"Error: Invalid schema definition for key '{key}'"

    return True, True


def validate_agent_response(raw_response: str, context: Optional[Dict[str, Any]] = None) -> Tuple[Optional[dict], Optional[str]]:
    """
    Validate agent JSON response. Assumes raw_response is a single dict.

    Args:
        raw_response (str): Raw JSON string from agent
        context (dict, optional): Game state context for advanced validation
            - known_data_map: dict mapping source_host -> list of {owner, id} dicts
            - known_services_map: dict mapping host -> list of service strings

    Returns:
        Tuple (validated_dict, None) on success or (None, error_message) on failure.
    """
    try:
        response = json.loads(raw_response)

        if not isinstance(response, dict):
            return None, "Error: Response must be a JSON object."

        action = response.get("action")
        if not action:
            return None, "Error: Missing 'action' field."

        schema_def = ACTION_SCHEMA.get(action)
        if not schema_def:
            return None, f"Error: Unknown action '{action}'."

        parameters = response.get("parameters")
        if not isinstance(parameters, dict):
            return None, "Error: 'parameters' must be a dictionary."

        # Action-specific pre-normalization
        if action == "ExfiltrateData":
            _normalize_exfiltrate_data(parameters, context=context)

        # Check required fields
        required_fields = schema_def.get("required", [])
        for field in required_fields:
            if field not in parameters:
                return None, f"Error: Missing required parameter '{field}' for action '{action}'."

        # Validate schema
        schema = schema_def.get("schema", {})
        is_valid, validation_error = validate_schema(parameters, schema)
        if not is_valid:
            return None, "Parameter validation failed: " + validation_error

        # Optional context-aware checks beyond pure schema
        if context and action == "ExploitService":
            try:
                host = parameters.get("target_host")
                svc = str(parameters.get("target_service", "")).lower().strip()
                ksm = context.get("known_services_map") or {}
                services = [str(s).lower() for s in ksm.get(host, [])]
                if services and svc not in services:
                    return None, (
                        f"Error: target_service '{parameters.get('target_service')}' not present "
                        f"for host '{host}' in known_services."
                    )
            except Exception:
                # On any failure, skip the extra check
                pass

        return response, None  # always return a 2-tuple

    except json.JSONDecodeError as e:
        return None, f"Error: Invalid JSON format. {e}"
