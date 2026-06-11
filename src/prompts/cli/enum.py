"""Prompts for enum module types."""

import enum

from beanapp import register


@register(enum.Enum)
def prompt_enum(
    message: str = "Please choose an option...",
    data_type: type = enum.Enum,
):
    """Prompt for a single enum member by name or value."""
    options = "\n - ".join([member.name for member in data_type])
    while True:
        selection = input(f"\n{message}\nOptions:{options}\n")

        for member in data_type:
            if selection.lower() == member.name.lower():
                return member

        print("Invalid selection. Please try again.")
