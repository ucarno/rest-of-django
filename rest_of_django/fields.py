import inspect
from decimal import Decimal
from functools import cache
from types import MappingProxyType
from typing import Pattern, Iterable, TypeVar, Callable, Type, Any, Optional

from django.core import validators as dj_validators
from django.core.exceptions import ValidationError
from django.db.models import Model

from django.utils.translation import gettext_lazy as _

import validators as _validators


T = TypeVar('T')
U = TypeVar('U')

T_DEFAULT = Optional[T | Callable[[], T]]
T_VALUE_MSG = T | tuple[T, str]

# todo: simplify regexes?
T_REGEXES = (
    str |                                # non-compiled regex
    Pattern |                            # compiled regex
    Iterable[Pattern | str] |            # iterable of compiled or non-compiled regexes
    Iterable[tuple[Pattern | str, str]]  # iterable of compiled or non-compiled regexes and their errors
)
T_VALIDATORS = Iterable[Callable[[Any], Any]]
T_DUNDER_CALL = Callable[[Optional[T]], None]

# use this instead of 'None' when 'None' is also a valid value
_UNDEFINED = object()

# empty immutable dict
_EMPTY_DICT = MappingProxyType({})


def get_value_and_error(v: T_VALUE_MSG[T], default_error: str = None) -> tuple[T, str | None]:
    if isinstance(v, Iterable) and not isinstance(v, str):
        val = tuple(v)
        val, error = val
        return val, error or default_error
    return v, default_error


def merge_serializer_errors(errors: list['SerializerError']):
    errors_dict = {}
    for err in errors:
        errors_dict.update(err.parse_data())  # todo: warning or exception on conflicts
    return SerializerError(errors_dict)


class SerializerError(ValidationError):
    def __init__(self, data: dict, message: str = 'Serializer error.', code: str = 'serializer_error'):
        super().__init__(message, code)
        self.data = data

    def parse_data(self, data: dict = None, only_first: bool = False, only_messages: bool = False) -> dict:
        data = data if data else self.data

        def errors_from_exception(error: ValidationError):
            _errors = error.error_list[0:(1 if only_first else None)]
            _errors_list = []
            for e in _errors:
                msg = e.message % (e.params or _EMPTY_DICT)
                _errors_list.append(msg if only_messages else {'code': e.code, 'message': msg})
            return _errors_list

        errors = {}
        for key, value in data.items():
            if isinstance(value, SerializerError):
                errors[key] = self.parse_data(value.data, only_first, only_messages)
            else:
                current_errors = sum([errors_from_exception(v) for v in value], [])
                errors[key] = current_errors[0] if only_first else current_errors
            continue
        return errors


# todo: more elegant error definition
class BaseField:
    ERROR_NULL = _('Field value can not be null.')
    ERROR_REQUIRED = _('This field is required.')
    ERROR_READ_ONLY = _('This field is read only.')
    ERROR_WRITE_ONLY = _('This field is write only')
    ERROR_TYPE = _('Field is in incorrect type.')  # todo: add list of supported types

    def __init__(
            self, *, source: str = None, getter: str = None, setter: str = None, default: T_DEFAULT[Any] = _UNDEFINED,
            allow_null: bool = False, is_required: bool = True, is_snippet: bool = True,
            read_only: bool = False, write_only: bool = False,
            validators: Iterable[Callable[[Any], Any]] = None, json_type: Type | tuple[type, ...] = None
    ):
        assert not (source and (getter or setter)), "'source' can not be combined with 'getter' or 'setter'"
        assert not (write_only and read_only), "Field can not be 'write_only' and 'read_only' at the same time"
        assert not (read_only and is_required), "Field can not be 'read_only' and 'is_required' at the same time"
        assert not (write_only and is_snippet), "Field can not be 'write_only' and 'is_snippet' at the same time"
        assert not (is_required and default != _UNDEFINED), "Field can not be 'is_required' and have a 'default' set"
        assert not (not allow_null and default is None), "'default' can not be 'None' if 'allow_null' is 'False'"

        self.source = source  # acts as getter and setter at the same time
        self.getter = getter  # model field name to get value
        self.setter = setter  # model field name to store value

        # list of django validators
        self.validators: list[Callable[[Any], Any]] = list(validators) if validators else []

        self.json_type: tuple[type] = json_type
        if self.json_type:
            self.json_type = (self.json_type,) if isinstance(self.json_type, type) else tuple(self.json_type)

        self.allow_null = allow_null
        self.is_required = is_required
        self.read_only = read_only
        self.write_only = write_only
        self.is_snippet = is_snippet

        # default value, can be a function with no mandatory arguments
        self.default: T_DEFAULT[Any] = default

    @staticmethod
    def to_python(json_value):
        """Prepares JSON data to be converted to Python"""
        return json_value

    @staticmethod
    def to_json(python_value):
        """Prepares Python data to be converted to JSON"""
        return python_value

    def is_valid_field_base(self, value) -> None:
        """Base validation of a field (type, 'allow_null', etc.)"""
        if value is None:
            if not self.allow_null:
                raise ValidationError([ValidationError(message=self.ERROR_NULL, code='null_not_allowed')])
            return

        if self.json_type and not isinstance(value, self.json_type):
            raise ValidationError([ValidationError(message=self.ERROR_TYPE, code='wrong_type')])

    def is_valid_field(self, value) -> None:
        """Validates a field"""
        self.is_valid_field_base(value)

        errors = []

        for validator in self.validators:
            try:
                validator(value)
            except ValidationError as e:
                errors.append(e)

        if len(errors) > 0 and isinstance(errors[0], SerializerError):
            if len(errors) == 1:
                raise errors[0]
            serializer_error = merge_serializer_errors([i for i in errors if isinstance(i, SerializerError)])
            raise serializer_error
        elif errors:
            raise ValidationError(errors)


class Serializer(BaseField):
    ERROR_FORBIDDEN_ELEMENTS = _('Mapping contains forbidden elements.')  # todo: specify which?

    def __init__(self, *, instance: Model = None, data: dict = None, is_partial: bool = False, **kwargs):
        assert not (kwargs and instance), "You can not use a serializer as a field with passed 'instance' to it"
        assert not (kwargs and data is not None), "You can not use a serializer as a field with passed 'data' to it"
        assert not (data and is_partial), "It seems you wanted to partially validate data you passed as 'data' argument. " \
                                          "If so, 'is_partial' must be passed into `.validate` method"

        super().__init__(json_type=dict, **kwargs)

        self.is_partial = is_partial

        self.instance = instance
        self.data = data

    def is_valid_field(self, value: dict, only_first: bool = False, is_partial: bool = False) -> None:
        """Acts as a validator of a serializer as a field"""
        self.is_valid_field_base(value)

        fields = self.get_fields()
        errors = {}

        for field_name, field in fields.items():
            field: BaseField

            if field_name in value:
                try:
                    field.is_valid_field(value[field_name])
                except SerializerError as error:
                    errors[field_name] = error
                except ValidationError as error:
                    errors[field_name] = errors.get(field_name, []) + [error]
            else:
                if field.is_required and not is_partial:
                    errors[field_name] = errors.get(field_name, []) + [ValidationError(field.ERROR_REQUIRED)]

        if errors:
            raise SerializerError(errors)

    def is_valid(self, *, is_partial: bool = False, raise_error: bool = False) -> bool:
        """Validates a serializer"""
        if self.data is None:
            raise ValueError("Could not validate, because 'data' was not provided")

        try:
            self.is_valid_field(self.data, is_partial=is_partial)
            return True
        except ValidationError as e:
            if raise_error:
                raise e
            return False

    @classmethod
    @cache
    def get_fields(cls) -> MappingProxyType[str, BaseField]:
        """Returns an immutable dict of fields in form of {field_name: Field}"""
        fields = inspect.getmembers(cls, lambda x: not inspect.isroutine(x) and isinstance(x, BaseField))
        return MappingProxyType(dict(fields))

    @classmethod
    def get_required_fields(cls) -> set[str]:
        """Returns a set of required field names"""
        fields = cls.get_fields()
        required_fields = {k for k, v in fields.items() if v.is_required}
        return required_fields

    @classmethod
    def get_snippet_fields(cls) -> set[str]:
        """Returns a set of snippet field names"""
        fields = cls.get_fields()
        snippet_fields = {k for k, v in fields.items() if v.is_snippet}
        return snippet_fields


class BooleanField(BaseField):
    def __init__(self, *, default: T_DEFAULT[bool] = _UNDEFINED, **kwargs):
        super().__init__(**kwargs, json_type=bool, default=default)


class CharField(BaseField):
    def __init__(
            self, *, min_length: int = None, max_length: int = None, regexes: T_REGEXES = None,
            default: T_DEFAULT[str] = _UNDEFINED, **kwargs
    ):
        super().__init__(**kwargs, json_type=str, default=default)
        self.default: T_DEFAULT[str]

        self.min_length = min_length
        self.max_length = max_length

        if self.min_length and self.max_length:
            assert self.min_length <= self.max_length, "'max_length' must be greater or equal to 'min_length'"

        if self.min_length:
            self.validators.append(dj_validators.MinLengthValidator(min_length))
        if self.max_length:
            self.validators.append(dj_validators.MaxLengthValidator(max_length))

        if regexes:
            if isinstance(regexes, str):
                regexes_and_errors = get_value_and_error(regexes),
            else:
                vals = tuple(get_value_and_error(val) for val in regexes)
                regexes_and_errors = vals

            for regex, error in regexes_and_errors:
                self.validators.append(dj_validators.RegexValidator(regex, error))


class NumericField(BaseField):
    def __init__(
            self, *, min_value: int = None, max_value: int = None,
            default: T_DEFAULT[int] = _UNDEFINED, **kwargs
    ):
        super().__init__(json_type=kwargs.pop('json_type', (int, float)), **kwargs, default=default)
        self.default: T_DEFAULT[int]

        self.min_value = min_value
        self.max_value = max_value

        if self.min_value and self.max_value:
            assert self.min_value <= self.max_value, "'max_value' must be greater or equal to 'min_value'"

        if self.min_value:
            self.validators.append(dj_validators.MinValueValidator(min_value))
        if self.max_value:
            self.validators.append(dj_validators.MaxValueValidator(max_value))


class IntegerField(NumericField):
    def __init__(self, *, default: T_DEFAULT[int] = _UNDEFINED, **kwargs):
        super().__init__(**kwargs, json_type=int, default=default)
        self.default: T_DEFAULT[int]


class FloatField(NumericField):
    def __init__(self, *, default: T_DEFAULT[float] = _UNDEFINED, **kwargs):
        super().__init__(**kwargs, json_type=float, default=default)
        self.default: T_DEFAULT[float]


class DecimalField(NumericField):
    def __init__(
            self, *, max_digits: int = None, decimal_places: int = None,
            default: T_DEFAULT[Decimal] = _UNDEFINED, **kwargs
    ):
        super().__init__(**kwargs, json_type=float, default=default)  # todo: cast float into decimal

        self.max_digits = max_digits
        self.decimal_places = decimal_places

        if self.max_digits or self.decimal_places:
            self.validators.append(dj_validators.DecimalValidator(self.max_digits, self.decimal_places))

    @staticmethod
    def to_json(python_value: Decimal):
        return float(python_value)

    @staticmethod
    def to_python(json_value: float):
        return Decimal(json_value)


class DictField(BaseField):
    def __init__(
            self, *, min_items: int, max_items: int, key_field: CharField = None, value_field: BaseField = None,
            default: T_DEFAULT[dict] = _UNDEFINED, **kwargs
    ):
        super().__init__(**kwargs, json_type=dict, default=default)
        self.default: T_DEFAULT[dict]

        self.min_items = min_items
        self.max_items = max_items

        if self.min_items and self.max_items:
            assert self.min_items <= self.max_items, "'max_items' must be greater or equal to 'min_items'"

        if self.min_items:
            self.validators.append(dj_validators.MinLengthValidator(self.min_items))  # todo: correct error
        if self.max_items:
            self.validators.append(dj_validators.MaxLengthValidator(self.max_items))  # todo: correct error

        self.key_field = key_field
        self.value_field = value_field

        if self.key_field:
            assert not self.key_field.allow_null, "Dict keys can not be 'None', please edit 'allow_null' argument"
            self.validators.append(_validators.DictFieldKeyValidator(self.key_field))

        if self.value_field:
            self.validators.append(_validators.DictFieldValueValidator(self.value_field))


class ListField(BaseField):
    def __init__(
            self, *, min_length=None, max_length=None, unique_items=False, item_field: BaseField = None,
            default: T_DEFAULT[list | tuple] = _UNDEFINED, **kwargs
    ):
        super().__init__(**kwargs, json_type=list, default=default)
        self.default: T_DEFAULT[list | tuple]

        self.min_length: Optional[int]
        self.max_length: Optional[int]
        self.unique_items: bool

        self.min_length = min_length
        self.max_length = max_length
        self.unique_items = unique_items

        if self.min_length and self.max_length:
            assert self.min_length <= self.max_length, "'max_length' must be greater or equal to 'min_length'"

        if self.min_length:
            self.validators.append(dj_validators.MinLengthValidator(self.min_length))  # todo: correct error
        if self.max_length:
            self.validators.append(dj_validators.MaxLengthValidator(self.max_length))  # todo: correct error
        if self.unique_items:
            self.validators.append(_validators.UniqueItemsValidator())

        self.item_field = item_field
        if self.item_field:
            self.validators.append(_validators.ListFieldItemValidator(self.item_field))
