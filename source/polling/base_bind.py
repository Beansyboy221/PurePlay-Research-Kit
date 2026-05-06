import pydantic
import typing


class Bind(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True)
    name: str
    poll_id: typing.Any
    registry: typing.ClassVar[list[typing.Self]] = []

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls.registry = []

    def __init__(self, name: str, poll_id: typing.Any, *args, **kwargs):
        super().__init__(name=name, poll_id=poll_id, *args, **kwargs)

    def model_post_init(self, __context: typing.Any) -> None:
        self.__class__.registry.append(self)
        if self.__class__ is not Bind:
            Bind.registry.append(self)
