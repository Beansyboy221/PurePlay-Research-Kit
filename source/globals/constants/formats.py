TIMESTAMP_FORMAT = '%b-%d-%Y_%I-%M%p'
BOOL_STRINGS = {
    **dict.fromkeys(
        iterable=['y',],
        value=True
    ),
    **dict.fromkeys(
        iterable=['n',], 
        value=False
    )
}