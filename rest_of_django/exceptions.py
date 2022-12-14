from typing import Union

from django.conf import settings
from django.core.exceptions import ValidationError


def _format_error(error: ValidationError):
    if settings.ROD_FORMAT == 'code':
        return error.code or 'invalid'

    elif settings.ROD_FORMAT == 'message':
        return error.message % (error.params or {})

    else:
        return {'code': error.code or 'invalid', 'message': error.message % (error.params or {})}


class SerializerError(Exception):
    def __init__(self, exceptions: list[Union['SerializerError', ValidationError], ...] | dict[str, 'SerializerError'] | ValidationError = None):
        self.exceptions = exceptions

    def __str__(self):
        return f'SerializerError<{self.parse_base()}>'

    def parse_base(self):
        if isinstance(self.exceptions, ValidationError):
            return {'base': _format_error(self.exceptions)}
        else:
            return {'fields': self.parse()}

    def parse(self) -> dict | list:
        if isinstance(self.exceptions, list):
            parsed = []
            for e in self.exceptions:
                if isinstance(e, SerializerError):
                    parsed.append(e.parse())
                elif isinstance(e, ValidationError):
                    parsed.append(_format_error(e))
            return parsed[0] if settings.ROD_ONLY_FIRST else parsed
        elif isinstance(self.exceptions, dict):
            parsed = {}
            for key, value in self.exceptions.items():
                parsed[key] = value.parse()
            return parsed
