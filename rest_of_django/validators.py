from __future__ import annotations

from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _

import fields


class UniqueItemsValidator:
    message = _('Array values must be unique.')
    code = 'unique_items'

    def __init__(self, message=None, code=None):
        if message is not None:
            self.message = message
        if code is not None:
            self.code = code

    def __call__(self, value):
        if len(value) != len(set(value)):
            raise ValidationError(self.message, code=self.code)


# todo: merge dict validators into one?
class DictFieldKeyValidator:
    code = 'dict_key'

    def __init__(self, field: fields.CharField, code=None):
        self.field = field
        if code is not None:
            self.code = code

    def __call__(self, value: dict):
        errors = {}
        for key in value.keys():
            try:
                self.field.is_valid_field(key)
            except ValidationError as error:
                errors[f'key:{key}'] = errors.get(key, []) + [error]

        if errors:
            raise fields.SerializerError(errors)


class DictFieldValueValidator:
    code = 'dict_value'

    def __init__(self, field: fields.BaseField, code=None):
        self.field = field
        if code is not None:
            self.code = code

    def __call__(self, value: dict):
        errors = {}
        for key, dict_value in value.items():
            try:
                self.field.is_valid_field(dict_value)
            except ValidationError as error:
                errors[key] = errors.get(key, []) + [error]

        if errors:
            raise fields.SerializerError(errors)


class ListFieldItemValidator:
    code = 'list_item'

    def __init__(self, field: fields.BaseField, code=None):
        self.field = field
        if code is not None:
            self.code = code

    def __call__(self, value: list):
        errors = {}
        for index, item in enumerate(value):
            try:
                self.field.is_valid_field(item)
            except fields.SerializerError as error:
                errors[str(index)] = error
            except ValidationError as error:
                errors[str(index)] = [error]

        if errors:
            raise fields.SerializerError(errors)
