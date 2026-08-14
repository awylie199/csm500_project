def is_not_empty(value: object) -> bool:
    """Check if a value is Empty (i.e. not Falsy)"""

    return isinstance(value, str) and bool(value.strip())
