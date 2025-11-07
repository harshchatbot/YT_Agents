```python
from typing import Any, Dict, List, Union

from pydantic import BaseModel, ValidationError


class DataValidator:
    """A class for validating data using Pydantic schemas and basic type checking.

    Attributes:
        None
    """

    def validate_schema(self, data: Dict[str, Any], schema: BaseModel) -> Union[BaseModel, List[Dict[str, Any]]]:
        """Validates data against a Pydantic schema.

        Args:
            data (Dict[str, Any]): The data to validate.
            schema (BaseModel): The Pydantic schema to use for validation.

        Returns:
            Union[BaseModel, List[Dict[str, Any]]]: Returns the validated data as a Pydantic model if successful.
            Returns a list of errors with associated input values if validation fails.
        """
        try:
            return schema(**data)
        except ValidationError as e:
            errors = []
            for error in e.errors():
                errors.append({
                    'loc': error['loc'],
                    'msg': error['msg'],
                    'type': error['type'],
                    'input': data.get(error['loc'][0]) if error['loc'] else data
                })
            return errors

    def check_type(self, value: Any, expected_type: type) -> bool:
        """Checks if a value is of the expected type.

        Args:
            value (Any): The value to check.
            expected_type (type): The expected data type.

        Returns:
            bool: True if the value is of the expected type, False otherwise.
        """
        return isinstance(value, expected_type)


if __name__ == "__main__":
    # Example Pydantic schema
    class User(BaseModel):
        id: int
        name: str
        signup_ts: Union[datetime, None] = None
        friends: List[int] = []

    from datetime import datetime

    # Example data
    data = {"id": 1, "name": "John Doe", "signup_ts": "2023-10-26T10:00:00", "friends": [2, 3]}
    invalid_data = {"id": "invalid", "name": "Jane Doe"}

    # Create a DataValidator instance
    validator = DataValidator()

    # Validate the data against the schema
    validated_data = validator.validate_schema(data, User)
    print(f"Validated data: {validated_data}\n")

    validated_invalid_data = validator.validate_schema(invalid_data, User)
    print(f"Validated invalid data: {validated_invalid_data}\n")

    # Check the type of a value
    is_integer = validator.check_type(10, int)
    print(f"Is 10 an integer? {is_integer}\n")

    is_string = validator.check_type("hello", str)
    print(f"Is \"hello\" a string? {is_string}\n")

    is_list = validator.check_type([1,2,3], list)
    print(f"Is [1,2,3] a list? {is_list}\n")

    is_bool = validator.check_type(True, bool)
    print(f"Is True a boolean? {is_bool}\n")
```