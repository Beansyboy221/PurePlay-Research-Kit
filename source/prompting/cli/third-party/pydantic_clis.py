import pydantic
import typing

from source.globals import logger
from .....prompt_utils.cli import cli_prompter

T = typing.TypeVar('T')

@cli_prompter.register(pydantic.BaseModel, 'Populate this Pydantic model.')
def prompt_for_pydantic_model(
        data_type: type[pydantic.BaseModel], 
        title: str, 
        message: str
    ) -> pydantic.BaseModel:
    field_dict = {}
    for field_name, field_info in data_type.model_fields.items():
        while True:
            field_value = cli_prompter.prompt_with_cli(
                data_type=field_info.annotation, 
                title=field_name, 
                message=field_info.description
            )
            if field_value is None and field_info.is_required():
                logger.warning('Field is required. Please try again.')
                continue
            break
        field_dict[field_name] = field_value
    return data_type.model_validate(field_dict)

# @cli_prompter.register(pydantic.PositiveInt, 'Please enter a positive integer.')
# @cli_prompter.register(pydantic.NonNegativeInt, 'Please enter a non-negative integer.')
# @cli_prompter.register(pydantic.NegativeInt, 'Please enter a negative integer.')
# @cli_prompter.register(pydantic.NonPositiveInt, 'Please enter a non-positive integer.')
# @cli_prompter.register(pydantic.PositiveFloat, 'Please enter a positive float.')
# @cli_prompter.register(pydantic.NonNegativeFloat, 'Please enter a non-negative float.')
# @cli_prompter.register(pydantic.NegativeFloat, 'Please enter a negative float.')
# @cli_prompter.register(pydantic.NonPositiveFloat, 'Please enter a non-positive float.')
# @cli_prompter.register(pydantic.FiniteFloat, 'Please enter a finite float.')
# @cli_prompter.register(pydantic.PastDate, 'Please enter a past date.')
# @cli_prompter.register(pydantic.FutureDate, 'Please enter a future date.')
# @cli_prompter.register(pydantic.PastDatetime, 'Please enter a past datetime.')
# @cli_prompter.register(pydantic.FutureDatetime, 'Please enter a future datetime.')
# @cli_prompter.register(pydantic.AwareDatetime, 'Please enter an aware datetime.')
# @cli_prompter.register(pydantic.NaiveDatetime, 'Please enter a naive datetime.')
# @cli_prompter.register(pydantic.NewPath, 'Please enter a new path.')
# @cli_prompter.register(pydantic.FilePath, 'Please enter a file path.')
# @cli_prompter.register(pydantic.DirectoryPath, 'Please enter a directory path.')
# @cli_prompter.register(pydantic.SocketPath, 'Please enter a socket path.')
# @cli_prompter.register(pydantic.IPvAnyAddress, 'Please enter an IPv4 or IPv6 address.')
# @cli_prompter.register(pydantic.IPvAnyInterface, 'Please enter an IPv4 or IPv6 interface.')
# @cli_prompter.register(pydantic.IPvAnyNetwork, 'Please enter an IPv4 or IPv6 network.')
# #@cli_prompter.register(pydantic.EmailStr, 'Please enter an email address.') # Requires email-validator package
# @cli_prompter.register(pydantic.Json, 'Please enter a JSON string.')
# @cli_prompter.register(pydantic.UUID1, 'Please enter a UUID1.')
# @cli_prompter.register(pydantic.UUID3, 'Please enter a UUID3.')
# @cli_prompter.register(pydantic.UUID4, 'Please enter a UUID4.')
# @cli_prompter.register(pydantic.UUID5, 'Please enter a UUID5.')
# @cli_prompter.register(pydantic.UUID6, 'Please enter a UUID6.')
# @cli_prompter.register(pydantic.UUID7, 'Please enter a UUID7.')
# @cli_prompter.register(pydantic.UUID8, 'Please enter a UUID8.')
@cli_prompter.register(pydantic.NameEmail, 'Please enter a name and email address.')
@cli_prompter.register(pydantic.AnyUrl, 'Please enter a URL.')
@cli_prompter.register(pydantic.HttpUrl, 'Please enter a HTTP URL.')
@cli_prompter.register(pydantic.AnyHttpUrl, 'Please enter a HTTP URL.')
@cli_prompter.register(pydantic.FileUrl, 'Please enter a file URL.')
@cli_prompter.register(pydantic.SecretStr, 'Please enter a secret string.')
@cli_prompter.register(pydantic.SecretBytes, 'Please enter a secret bytes.')
@cli_prompter.register(pydantic.PostgresDsn, 'Please enter a PostgreSQL DSN.')
@cli_prompter.register(pydantic.RedisDsn, 'Please enter a Redis DSN.')
@cli_prompter.register(pydantic.MongoDsn, 'Please enter a MongoDB DSN.')
@cli_prompter.register(pydantic.AmqpDsn, 'Please enter an AMQP DSN.')
@cli_prompter.register(pydantic.KafkaDsn, 'Please enter a Kafka DSN.')
def prompt_for_pydantic_type(
        data_type: T, 
        title: str, 
        message: str = 'Please enter a pydantic type.'
    ) -> T:
    try:
        return pydantic.TypeAdapter(data_type).validate_python(
            cli_prompter.prompt_with_cli(str, title, message)
        )
    except pydantic.ValidationError as e:
        logger.warning(f'Invalid input: {e}. Please try again.')
        return prompt_for_pydantic_type(data_type, title, message)