import pydantic
import typing

from misc import logging_utils
from prompting.tkinter import helpers

logger = logging_utils.get_logger()

T = typing.TypeVar('T')

@helpers.register(pydantic.BaseModel, 'Populate this Pydantic model.')
def prompt_for_pydantic_model(
        data_type: type[pydantic.BaseModel], 
        title: str, 
        message: str
    ) -> pydantic.BaseModel:
    field_dict = {}
    for field_name, field_info in data_type.model_fields.items():
        while True:
            field_value = helpers.create_prompt(
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

@helpers.register(pydantic.PositiveInt, 'Please enter a positive integer.')
@helpers.register(pydantic.NonNegativeInt, 'Please enter a non-negative integer.')
@helpers.register(pydantic.NegativeInt, 'Please enter a negative integer.')
@helpers.register(pydantic.NonPositiveInt, 'Please enter a non-positive integer.')
@helpers.register(pydantic.PositiveFloat, 'Please enter a positive float.')
@helpers.register(pydantic.NonNegativeFloat, 'Please enter a non-negative float.')
@helpers.register(pydantic.NegativeFloat, 'Please enter a negative float.')
@helpers.register(pydantic.NonPositiveFloat, 'Please enter a non-positive float.')
@helpers.register(pydantic.FiniteFloat, 'Please enter a finite float.')
@helpers.register(pydantic.PastDate, 'Please enter a past date.')
@helpers.register(pydantic.FutureDate, 'Please enter a future date.')
@helpers.register(pydantic.PastDatetime, 'Please enter a past datetime.')
@helpers.register(pydantic.FutureDatetime, 'Please enter a future datetime.')
@helpers.register(pydantic.AwareDatetime, 'Please enter an aware datetime.')
@helpers.register(pydantic.NaiveDatetime, 'Please enter a naive datetime.')
@helpers.register(pydantic.NewPath, 'Please enter a new path.')
@helpers.register(pydantic.FilePath, 'Please enter a file path.')
@helpers.register(pydantic.DirectoryPath, 'Please enter a directory path.')
@helpers.register(pydantic.SocketPath, 'Please enter a socket path.')
@helpers.register(pydantic.IPvAnyAddress, 'Please enter an IPv4 or IPv6 address.')
@helpers.register(pydantic.IPvAnyInterface, 'Please enter an IPv4 or IPv6 interface.')
@helpers.register(pydantic.IPvAnyNetwork, 'Please enter an IPv4 or IPv6 network.')
#@helpers.register(pydantic.EmailStr, 'Please enter an email address.') # Requires email-validator package
@helpers.register(pydantic.Json, 'Please enter a JSON string.')
@helpers.register(pydantic.UUID1, 'Please enter a UUID1.')
@helpers.register(pydantic.UUID3, 'Please enter a UUID3.')
@helpers.register(pydantic.UUID4, 'Please enter a UUID4.')
@helpers.register(pydantic.UUID5, 'Please enter a UUID5.')
@helpers.register(pydantic.UUID6, 'Please enter a UUID6.')
@helpers.register(pydantic.UUID7, 'Please enter a UUID7.')
@helpers.register(pydantic.UUID8, 'Please enter a UUID8.')
@helpers.register(pydantic.NameEmail, 'Please enter a name and email address.')
@helpers.register(pydantic.AnyUrl, 'Please enter a URL.')
@helpers.register(pydantic.HttpUrl, 'Please enter a HTTP URL.')
@helpers.register(pydantic.AnyHttpUrl, 'Please enter a HTTP URL.')
@helpers.register(pydantic.FileUrl, 'Please enter a file URL.')
@helpers.register(pydantic.SecretStr, 'Please enter a secret string.')
@helpers.register(pydantic.SecretBytes, 'Please enter a secret bytes.')
@helpers.register(pydantic.PostgresDsn, 'Please enter a PostgreSQL DSN.')
@helpers.register(pydantic.RedisDsn, 'Please enter a Redis DSN.')
@helpers.register(pydantic.MongoDsn, 'Please enter a MongoDB DSN.')
@helpers.register(pydantic.AmqpDsn, 'Please enter an AMQP DSN.')
@helpers.register(pydantic.KafkaDsn, 'Please enter a Kafka DSN.')
def prompt_for_pydantic_type(
        data_type: T, 
        title: str, 
        message: str = 'Please enter a pydantic type.'
    ) -> T:
    try:
        return pydantic.TypeAdapter(data_type).validate_python(
            helpers.create_prompt(str, title, message)
        )
    except pydantic.ValidationError as e:
        logger.warning(f'Invalid input: {e}. Please try again.')
        return prompt_for_pydantic_type(data_type, title, message)