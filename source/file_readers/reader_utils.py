from . import base_reader

# Considerations:
# Ensure .get results in type being highlighted
# The order in which this is imported matters
# Should this be defined when it is needed or at the start of the program?

AVAILABLE_READERS = dict[type, type[base_reader.BaseReader]]
'''
A registry of all loaded readers.
Use this to dynamically find readers.
Key: data-type to read
Value: reader class
'''

def register_reader(reader_class: type[base_reader.BaseReader]) -> None:
    AVAILABLE_READERS[reader_class.data_type] = reader_class
base_reader.BaseReader.on_init.connect(register_reader)