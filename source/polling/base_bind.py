import pydantic
import typing

class Bind(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True)
    id: typing.Any

    def __init__(self, id, **kwargs):
        super().__init__(id=id, **kwargs)