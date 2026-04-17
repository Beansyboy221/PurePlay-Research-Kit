import pydantic

from source.misc import validated_paths

class TestConfig(pydantic.BaseModel):
    '''Fields expected for test mode to work properly.'''
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for field_name, field_info in cls.model_fields.items():
            if field_info.validation_alias is None:
                field_info.validation_alias = pydantic.AliasPath('test', field_name)
        cls.model_rebuild(force=True)
    
    save_dir: pydantic.DirectoryPath = pydantic.Field(default='reports')
    '''The directory to save the output reports to.'''

    model_file: validated_paths.CheckpointPath
    '''The path to the model file.'''

    testing_file_dir: pydantic.DirectoryPath = pydantic.Field(default='data\\test')
    '''The directory containing the testing files.'''