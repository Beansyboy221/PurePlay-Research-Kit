"""Prompts for enum module types."""

import enum

from beanapp import register


@register(enum.Enum)
def prompt_enum(data_type: type, message: str):
    """Prompt for a single enum member by name or value."""
    options = ", ".join([f"{member.name}" for member in data_type])
    default = f"Choose one: ({options}):"
    while True:
        value = input(f"\n{message or default}\n").strip()

        for member in data_type:
            if value.lower() == member.name.lower():
                return member

        print(f"Invalid selection. Please choose from: {options}")


@register(enum.Flag)
def prompt_flag(data_type: type, message: str):
    """Prompt for one or more flag members (comma-separated)."""
    options = ", ".join([f"{member.name}" for member in data_type])
    default = f"Select one or more, separated by commas ({options}):"
    while True:
        value = input(f"\n{message or default}\n").strip()
        if not value:
            return data_type(0)

        parts = [p.strip() for p in value.split(",")]
        result = None

        try:
            for part in parts:
                member = None

                for name, m in data_type.__members__.items():
                    if part.lower() == name.lower():
                        member = m
                        break

                if member is None:
                    try:
                        member = data_type(int(part))
                    except ValueError:
                        raise ValueError(f"'{part}' is not a valid flag.")

                if result is None:
                    result = member
                else:
                    result |= member
            return result
        except ValueError as e:
            print(f"Invalid input: {e}")
