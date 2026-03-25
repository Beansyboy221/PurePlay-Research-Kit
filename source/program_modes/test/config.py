import pydantic

class ModeConfig(pydantic.BaseModel):
    '''Fields expected for test mode to work properly.'''

    save_dir: pydantic.DirectoryPath = pydantic.Field(
        default='reports',
        description='The directory to save the output reports to.',
        validation_alias=pydantic.AliasPath('test', 'save_dir')
    )
    '''The directory to save the output reports to.'''

    model_file: pydantic.FilePath = pydantic.Field(
        description='The path to the model file.',
        validation_alias=pydantic.AliasPath('test', 'model_file'),
        json_schema_extra={'file_types': [('Checkpoint Files', '*.ckpt'),]}
    )
    '''The path to the model file.'''

    testing_file_dir: pydantic.DirectoryPath = pydantic.Field(
        default='data\\test',
        description='The directory containing the testing files.',
        validation_alias=pydantic.AliasPath('test', 'testing_file_dir')
    )
    '''The directory containing the testing files.'''