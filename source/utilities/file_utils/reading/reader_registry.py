import pathlib

from reading import base_reader

READERS: dict[str, base_reader.FileReader] = [] 
'''A mapping of extensions to their associated readers.'''

def get_supported_extensions(output_type: type) -> set[str]:
    '''Returns a set of supported extensions for the given output type.'''
    return {
        extension 
        for reader_class in READERS.values()
        for extension in reader_class.extensions
        if reader_class.get_read_method(output_type) is not None
    }

def register(cls: type[base_reader.FileReader]) -> type[base_reader.FileReader]:
    '''Decorator that registers a reader by its extensions.'''
    for extension in cls.extensions:
        READERS[extension] = cls
    return cls

def get_reader(file_path: str) -> base_reader.FileReader:
    '''Returns an instance of the appropriate reader for the given file path.'''
    extension = pathlib.Path(file_path).suffix.lower()
    if extension not in READERS:
        raise ValueError(f'No reader registered for extension: {extension}')
    return READERS[extension]()