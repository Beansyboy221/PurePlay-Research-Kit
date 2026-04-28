import pydantic

from misc import validated_types
import modes

class TestConfig(modes.ModeConfig):
    '''Fields expected for test mode to work properly.'''
    save_dir: pydantic.DirectoryPath = pydantic.Field(default='reports')
    '''The directory to save the output reports to.'''

    model_file: validated_types.CheckpointPath
    '''The path to the model file.'''

    testing_file_dir: pydantic.DirectoryPath = pydantic.Field(default='data\\test')
    '''The directory containing the testing files.'''