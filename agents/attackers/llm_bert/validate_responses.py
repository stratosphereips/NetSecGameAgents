import json

# Define schema for each action
ACTION_SCHEMA = {
    "ExfiltrateData": {
        "required": ["target_host", "source_host", "data"],
        "schema": {
            "target_host": str,
            "source_host": str,
            "data": {
                "required": ["owner", "id"],
                "schema": {
                    "owner": str,
                    "id": str
                }
            }
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
    "FindServices": {
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


def validate_schema(data: dict, schema: dict) -> tuple[bool, str | bool]:
    """Recursively validate that data matches the schema."""
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


def validate_agent_response(raw_response: str) -> tuple[dict | None, str | None]:
    """Validate agent JSON response. Assumes raw_response is a single dict.

    Args:
        raw_response (str): Raw JSON string from agent

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

        return response, None  # ✅ always return a 2-tuple

    except json.JSONDecodeError as e:
        return None, f"Error: Invalid JSON format. {e}"
