from . import base_writer

# Considerations:
# Ensure .get results in type being highlighted
# The order in which this is imported matters
# Should this be defined when it is needed or at the start of the program?

AVAILABLE_WRITERS = dict[type, type[base_writer.BaseWriter]]
'''
A registry of all loaded writers.
Use this to dynamically find writers.
Key: data-type to write
Value: writer class
'''

def register_writer(writer_class: type[base_writer.BaseWriter]) -> None:
    AVAILABLE_WRITERS[writer_class.data_type] = writer_class
base_writer.BaseWriter.on_init.connect(register_writer)