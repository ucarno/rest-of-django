from django.core.exceptions import ValidationError


T_EXCEPTION = 'SerializerTreeError' | ValidationError


class SerializerError(Exception):
    def __init__(self, exceptions: list[T_EXCEPTION] | dict[str, T_EXCEPTION | list[T_EXCEPTION]] = None):
        self.exceptions = exceptions

    def __str__(self):
        return str(self.exceptions)  # todo
