import inspect
import re
from abc import abstractmethod
import datetime
from decimal import Decimal
from types import MappingProxyType
from typing import Pattern, Iterable, TypeVar, Callable, Any, Optional
from uuid import UUID

from django.conf import settings
from django.core import validators as dj_validators
from django.core.exceptions import ValidationError
from django.core.validators import EmailValidator, URLValidator, MinValueValidator, MaxValueValidator
from django.db.models import Model
from django.utils.dateparse import parse_date, parse_time, parse_datetime, parse_duration
from django.utils.translation import gettext_lazy as _

from rest_of_django.exceptions import SerializerError
from rest_of_django.utils import timedelta_to_iso8601
from rest_of_django.validators import ChoiceValidator, ipv4_address_validator, ipv6_address_validator, \
    ipv46_address_validator

T = TypeVar('T')
U = TypeVar('U')

T_DEFAULT = Optional[T | Callable[[], T]]
T_VALUE = T_DEFAULT

T_REGEX = str | Pattern
T_REGEXES = (
    T_REGEX,                                           # compiled or non-compiled regex
    Iterable[T_REGEX] |                                # iterable of compiled or non-compiled regexes
    Iterable[tuple[T_REGEX, Optional[str]] | T_REGEX]  # iterable of compiled or non-compiled regexes and their errors
)
T_VALIDATORS = Iterable[Callable[[Any], Any]]
T_DUNDER_CALL = Callable[[Optional[T]], None]

T_CHOICES = tuple[T | tuple[T, T | Callable[[], T]], ...]

# use this instead of 'None' when 'None' is also a valid value
_UNDEFINED = object()

# empty immutable dict
_EMPTY_DICT = MappingProxyType({})


JSON_TYPES = (str, int, float, bool, dict, list)
T_JSON_BASIC = type[str | int | float | bool]
T_JSON = T_JSON_BASIC | type[dict | list]


def patch_validator_messages(validator, **messages):
    """Patches validator's messages dict without modifying it globally"""
    validator.messages = {**validator.messages, **messages}
    return validator


def patch_validator(validator: dj_validators.BaseValidator, code: str = None):
    """Patches validator's code"""
    validator.code = code
    return validator


class BaseField:
    default_errors = {
        'null_not_allowed': _('Field value can not be null.'),
        'required': _('This field is required.'),
        'read_only': _('This field is read only.'),
        'write_only': _('This field is write only'),
        'type': _('Incorrect type. Expected \'%(expected_type)s\', but got \'%(received_type)s\'.'),
    }

    def __init__(
            self, *, source: str = None, getter: str = None, setter: str = None, default: T_DEFAULT[Any] = _UNDEFINED,
            allow_null: bool = False, required: bool = True, snippet: bool = True,
            read_only: bool = False, write_only: bool = False, json_type: T_JSON | tuple[T_JSON, ...] = None,
            validators: Iterable[Callable[[Any], Any]] = None, errors: dict = None
    ):
        """
        :param source: field name in a Django model
        :param getter: property or a method in a Django model to retrieve field value
        :param setter: method in a Django model to set data
        :param default: any value or a method, which takes no arguments and returns a value
        :param allow_null: whether this field can be `null`
        :param required: whether this field must be provided
        :param snippet: whether this field must be included by default in GET requests
        :param read_only: whether this field is for read only
        :param write_only: whether this field is for write only
        :param json_type: expected type or types parsed from JSON string
        :param validators: list of validators
        :param errors: error messages' overwrites
        """

        assert not (source and (getter or setter)), "'source' can not be combined with 'getter' or 'setter'"
        assert not (write_only and read_only), "Field can not be 'write_only' and 'read_only' at the same time"
        assert not (read_only and required), "Field can not be 'read_only' and 'required' at the same time"
        assert not (write_only and snippet), "Field can not be 'write_only' and 'snippet' at the same time"
        assert not (required and default != _UNDEFINED), "Field can not be 'required' and have a 'default'"
        assert not (not allow_null and default is None), "'default' can not be 'None' if 'allow_null' is 'False'"

        self.errors = {**self.default_errors, **(errors or {})}

        self.source = source  # acts as getter and setter at the same time

        # list of django validators
        self.validators: list[Callable[[Any], Any]] = list(validators) if validators else []

        self.json_type: tuple[T_JSON] = json_type
        if self.json_type:
            self.json_type = (self.json_type,) if isinstance(self.json_type, type) else tuple(set(self.json_type))
            assert len(set(self.json_type) - set(JSON_TYPES)) == 0, \
                   f"Valid 'json_type' values are {' | '.join([i.__class__.__name__ for i in JSON_TYPES])}"

        self.allow_null = allow_null
        self.required = required
        self.read_only = read_only
        self.write_only = write_only
        self.snippet = snippet

        # default value, can be a function with no mandatory arguments
        self.default: T_DEFAULT[Any] = default

    def to_python(self, json_value):
        """
        Converts JSON value to Python
        :raises: ValidationError: if there is an error on conversion
        """
        return json_value

    def to_json(self, python_value):
        """Converts Python value to JSON"""
        return python_value

    def validate_field_raw(self, value):
        """
        Validates raw value from parsed JSON before conversion and converts it to Python value for further validation
        :raises: ValidationError: if raw field is invalid
        :return: Converted Python value
        """
        if value is None:
            if not self.allow_null:
                raise ValidationError(self.errors['null_not_allowed'], 'null_not_allowed')
            return

        if self.json_type and not isinstance(value, self.json_type):
            raise ValidationError(self.errors['type'], 'type', {
                'received_type': type(value).__name__,
                'expected_type': ' | '.join([t.__name__ for t in self.json_type])
            })

        return self.to_python(value)

    def validate_field(self, value):
        """
        Validates converted Python value
        :raises: SerializerError: with list of exceptions if value is invalid
        :return: Converted value
        """

        errors = []
        try:
            converted = self.validate_field_raw(value)
        except ValidationError as e:
            errors.append(e)
            raise SerializerError(errors)

        for validator in self.validators:
            try:
                validator(converted)
            except ValidationError as e:
                errors.append(e)
                if settings.ROD_ONLY_FIRST:
                    break

        if len(errors) > 0:
            raise SerializerError(errors)

        return converted


class Serializer(BaseField):
    default_errors = {
        **BaseField.default_errors,
        'unexpected_keys': _('Object contains unexpected keys: %(unexpected_keys)s.')
    }

    def __init__(self, *, instance: Model = None, data: dict = None, **kwargs):
        """
        :param instance: Instance of a Django model
        :param is_partial: Specifies partial validation (e.g. for PATCH request)
        """
        assert not (kwargs and instance), "You can not use a serializer as a field with passed 'instance' to it"
        assert not (kwargs and data is not None), "You can not use a serializer as a field with passed 'data' to it"

        super().__init__(json_type=dict, **kwargs)

        self.instance = instance
        self.data = data

        self._validated_data = None

    @property
    def validated_data(self) -> dict:
        """
        Converted data that is available only after validation
        :raise: ValueError: if `is_valid() was not called earlier`
        """
        if not self._validated_data:
            raise ValueError('This function can be called only after data is validated using `is_valid()` method.')
        return self._validated_data

    def validate_field(self, value: dict, is_partial: bool = False, context: dict = None) -> dict:
        """Validates serializer as a field"""
        try:
            value = self.validate_field_raw(value)
        except ValidationError as e:
            raise SerializerError([e])

        contextual_validators = self.get_contextual_validators()

        errors = {}
        converted = {}

        for field_name, field in self.get_fields().items():
            field: BaseField

            if field_name in value:
                if field.read_only:
                    errors[field_name] = SerializerError([
                        ValidationError(message=field.errors['read_only'], code='read_only')
                    ])
                    continue

                try:
                    kwargs = {'value': value[field_name]}
                    if isinstance(field, Serializer):
                        kwargs['context'] = context
                    converted[field_name] = field.validate_field(**kwargs)
                except SerializerError as error:
                    errors[field_name] = error
                    continue

                if field_name in contextual_validators:
                    validator = contextual_validators[field_name]
                    args = set(inspect.signature(validator).parameters.keys())
                    is_static = isinstance(inspect.getattr_static(self, f'validate_{field_name}'), staticmethod)
                    requires_context = (len(args) > 2) if is_static else (len(args) > 1)

                    try:
                        if requires_context:
                            validator(converted[field_name], context or {})
                        else:
                            validator(converted[field_name])
                    except ValidationError as e:
                        converted.pop(field_name, None)
                        errors[field_name] = SerializerError([e])

            else:
                if field.default:
                    converted[field_name] = field.default() if callable(field.default) else field.default
                elif field.required and not is_partial and not field.read_only:
                    errors[field_name] = SerializerError([ValidationError(field.errors['required'], code='required')])

        if errors:
            raise SerializerError(errors)

        validate_together = getattr(self, 'validate')
        if not getattr(validate_together, '__isabstractmethod__', False):
            try:
                validate_together(data=converted, is_partial=is_partial, context=context)
            except ValidationError as e:
                raise SerializerError(e)

        return converted

    def is_valid(self, *, is_partial: bool = False, raise_error: bool = False, context: dict = None) -> bool:
        """Validates a serializer"""
        if self.data is None:
            raise ValueError("Could not validate, because 'data' was not provided")

        try:
            self.validate_field(self.data, is_partial=is_partial, context=context)
            return True
        except SerializerError as e:
            if raise_error:
                raise e
            return False

    @abstractmethod
    def validate(self, data: dict, is_partial: bool, context: dict):
        raise NotImplementedError(f"'validate' method is not implemented in '{self.__class__.__name__}' class.")

    @classmethod
    # @cache
    def get_fields(cls) -> MappingProxyType[str, BaseField]:
        """Returns an immutable dict of fields in form of {field_name: Field}"""
        fields = inspect.getmembers(cls, lambda x: not inspect.isroutine(x) and isinstance(x, BaseField))
        return MappingProxyType(dict(fields))

    @classmethod
    def get_required_fields(cls) -> set[str]:
        """Returns a set of required field names"""
        fields = cls.get_fields()
        required_fields = {k for k, v in fields.items() if v.required}
        return required_fields

    @classmethod
    def get_snippet_fields(cls) -> set[str]:
        """Returns a set of snippet field names"""
        fields = cls.get_fields()
        snippet_fields = {k for k, v in fields.items() if v.snippet}
        return snippet_fields

    @classmethod
    def get_contextual_validators(cls) -> dict[str, Callable[[Any, dict], None] | Callable[[Any], None]]:
        fields = cls.get_fields()

        validators = {}
        for field_name in fields.keys():
            validator_name = f'validate_{field_name}'
            if hasattr(cls, validator_name):
                validator = getattr(cls, validator_name)
                if callable(validator):
                    validators[field_name] = validator

        return validators


class BooleanField(BaseField):
    def __init__(self, *, default: T_DEFAULT[bool] = _UNDEFINED, **kwargs):
        super().__init__(json_type=bool, default=default, **kwargs)


class CharField(BaseField):
    default_errors = {
        **BaseField.default_errors,
        'min_length': _('Value must be at least %(min_length)s characters long.'),
        'max_length': _('Value can not exceed %(max_length)s characters.'),
        'invalid': _('Invalid value.'),
        'invalid_choice': ChoiceValidator.message
    }

    def __init__(
            self, *, min_length: int = None, max_length: int = None, regexes: T_REGEXES = None,
            choices: T_CHOICES[str] = None, default: T_DEFAULT[str] = _UNDEFINED, **kwargs
    ):
        """
        :param min_length: minimum length of a string
        :param max_length: maximum length of a string
        :param regexes: single regular expression or list of regular expressions in form of [re1, re2, ...] or [(re1, error1), (re2, error2), ...]
        :param choices: a tuple of strings or a tuple of two-sized tuples of strings in form of ((A, B), (C, D), ...)
        :param default:
        :param kwargs:
        """

        super().__init__(json_type=str, default=default, **kwargs)

        assert not (any((min_length, max_length, choices, kwargs.get('validators'))) and choices), \
               "'choices' can not be specified with 'min_length', 'max_length', 'choices' or 'validators'"
        assert not (choices and len(choices) == 0), "'choices' must contain at least 1 choice"

        self.min_length = min_length
        self.max_length = max_length

        if self.min_length and self.max_length:
            assert self.min_length <= self.max_length, "'max_length' must be greater or equal to 'min_length'"

        if self.min_length:
            self.validators.append(dj_validators.MinLengthValidator(min_length, self.errors['min_length']))
        if self.max_length:
            self.validators.append(dj_validators.MaxLengthValidator(max_length, self.errors['max_length']))

        if regexes:
            if isinstance(regexes, (str, Pattern)):
                self.validators.append(dj_validators.RegexValidator(
                    regexes,
                    message=self.errors['regex'],
                    code='regex',
                ))
            else:
                regexes = tuple(regexes)
                if len(regexes) == 0:
                    return

                for r in regexes:
                    regex, error = (r, self.errors['regex']) if isinstance(r, (str, Pattern)) else r
                    if not error:
                        error = self.errors['regex']

                    self.validators.append(dj_validators.RegexValidator(
                        regex,
                        message=error,
                        code='invalid',
                    ))

        if choices:
            if not isinstance(choices[0], str):
                choices = tuple(i for i, _ in choices)
            self.validators.append(ChoiceValidator(choices, message=self.errors['invalid_choice']))


class NumericField(BaseField):
    default_errors = {
        **BaseField.default_errors,
        'min_value': dj_validators.MinValueValidator.message,
        'max_value': dj_validators.MaxValueValidator.message,
    }

    def __init__(
            self, *, min_value: int = None, max_value: int = None,
            default: T_DEFAULT[int] = _UNDEFINED, **kwargs
    ):

        super().__init__(json_type=kwargs.pop('json_type', (int, float)), default=default, **kwargs)

        self.min_value = min_value
        self.max_value = max_value

        if self.min_value and self.max_value:
            assert self.min_value <= self.max_value, "'max_value' must be greater or equal to 'min_value'"

        if self.min_value:
            self.validators.append(dj_validators.MinValueValidator(min_value, self.errors['min_value']))
        if self.max_value:
            self.validators.append(dj_validators.MaxValueValidator(max_value, self.errors['max_value']))


class IntegerField(NumericField):
    def __init__(self, *, default: T_DEFAULT[int] = _UNDEFINED, **kwargs):
        super().__init__(json_type=int, default=default, **kwargs)


class FloatField(NumericField):
    def __init__(self, *, default: T_DEFAULT[float] = _UNDEFINED, **kwargs):
        super().__init__(json_type=float, default=default, **kwargs)


class DecimalField(NumericField):
    errors = {
        **NumericField.default_errors,
        'max_digits': dj_validators.DecimalValidator.messages['max_digits'],
        'max_decimal_places': dj_validators.DecimalValidator.messages['max_decimal_places'],
        'max_whole_digits': dj_validators.DecimalValidator.messages['max_whole_digits']
    }

    def __init__(
            self, *, max_digits: int = None, max_decimal_places: int = None,
            default: T_DEFAULT[Decimal] = _UNDEFINED, **kwargs
    ):

        super().__init__(json_type=float, default=default, **kwargs)

        self.max_digits = max_digits
        self.max_decimal_places = max_decimal_places

        if self.max_digits or self.max_decimal_places:
            assert self.max_digits and self.max_decimal_places, \
                "'max_digits' and 'max_decimal_places' should be defined both or not defined at all."
            validator = dj_validators.DecimalValidator(self.max_digits, self.max_decimal_places)
            patch_validator_messages(
                validator,
                max_digits=self.errors['max_digits'],
                max_decimal_places=self.errors['max_decimal_places'],
                max_whole_digits=self.errors['max_whole_digits']
            )

            self.validators.append(dj_validators.DecimalValidator(self.max_digits, self.max_decimal_places))

    def to_json(self, python_value: Decimal) -> float:
        return float(python_value)

    def to_python(self, json_value: float):
        return Decimal(json_value)


class DictField(BaseField):
    default_errors = {
        **BaseField.default_errors,
        'min_items': _('Object does not contain enough items, minimum item count is %(limit_value)d.'),
        'max_items': _('Object contains too many items, maximum item count is %(limit_value)d.'),
    }

    def __init__(
            self, *, min_items: int, max_items: int, key_field: CharField = None, value_field: BaseField = None,
            value_json_type: T_JSON | tuple[T_JSON, ...],
            default: T_DEFAULT[dict] = _UNDEFINED, **kwargs
    ):
        """
        :param min_items: minimum item count in a dictionary
        :param max_items: maximum item count in a dictionary
        :param key_field: a CharField that would be used to validate dictionary's key
        :param value_field: a CharField that would be used to validate dictionary's value
        :param value_json_type: value type that is expected from values in JSON
        """

        super().__init__(json_type=dict, default=default, **kwargs)

        assert not (value_field and value_json_type), "'value_field' can not be combined with 'value_json_type'"

        self.min_items = min_items
        self.max_items = max_items

        if self.min_items and self.max_items:
            assert self.min_items <= self.max_items, "'max_items' must be greater or equal to 'min_items'"

        if self.min_items:
            self.validators.append(patch_validator(
                dj_validators.MinLengthValidator(self.min_items, self.errors['min_items']),
                code='min_items'
            ))
        if self.max_items:
            self.validators.append(patch_validator(
                dj_validators.MaxLengthValidator(self.max_items, self.errors['max_items']),
                code='max_items'
            ))

        self.key_field = key_field
        self.value_field = value_field

        if value_json_type:
            self.value_field = BaseField(json_type=value_json_type)

        assert not (self.key_field and self.key_field.allow_null), \
               "Dict keys can not be null, edit 'allow_null' argument"

    def validate_field(self, value: dict) -> dict:
        """
        Validates dict field
        :raise SerializerError:
        """
        try:
            value: dict = self.validate_field_raw(value)
        except ValidationError as e:
            raise SerializerError([e])

        errors = {}
        converted = {}

        for item_key, item_value in value.items():
            converted_key = None
            converted_value = _UNDEFINED

            if self.key_field:
                try:
                    converted_key = self.key_field.validate_field(item_key)
                except SerializerError as e:
                    errors[f'{item_key}:key'] = e
            else:
                converted_key = item_key

            if self.value_field:
                try:
                    converted_value = self.value_field.validate_field(item_value)
                except SerializerError as e:
                    errors[item_key] = e
            else:
                converted_value = item_value

            if converted_key and converted_value != _UNDEFINED:
                converted[converted_key] = converted_value

        if errors:
            raise SerializerError(errors)

        errors = []
        for validator in self.validators:
            try:
                validator(value)
            except ValidationError as e:
                errors.append(e)
                if settings.ROD_ONLY_FIRST:
                    break

        if errors:
            raise SerializerError(errors)

        return converted


class ListField(BaseField):
    default_errors = {
        **BaseField.default_errors,
        'unique_items': _('Array values must be unique.'),
        'min_length': _('Array should contain at least %(limit_value)d values.'),
        'max_length': _('Array should not contain more than %(limit_value)d values.'),
    }

    def __init__(
            self, *, min_length=None, max_length=None, unique_items=False, item_field: BaseField = None,
            item_json_type: T_JSON_BASIC | tuple[T_JSON_BASIC, ...] = None,
            default: T_DEFAULT[list | tuple] = _UNDEFINED, **kwargs
    ):
        """
        :param min_length: minimum length of a list
        :param max_length: maximum length of a list
        :param unique_items: whether list items should be validated for uniqueness
        :param item_field: field that would be used to validate list's values
        :param item_json_type: type of value expected from a list
        """

        super().__init__(json_type=list, default=default, **kwargs)

        assert not (unique_items and isinstance(item_field, (ListField, DictField, Serializer))), \
            f"Can not combine 'unique_items' with 'item_field={item_field.__class__.__name__}(...)', " \
            f"because type of an item field must be hashable"
        assert not (item_field and item_json_type), "'item_field' can not be combined with 'item_json_type'"

        self.min_length: Optional[int] = min_length
        self.max_length: Optional[int] = max_length
        self.unique_items: bool = unique_items

        self.item_field = item_field
        if item_json_type:
            self.item_field = BaseField(json_type=item_json_type)

        if self.min_length and self.max_length:
            assert self.min_length <= self.max_length, "'max_length' must be greater or equal to 'min_length'"

        if self.min_length:
            self.validators.append(dj_validators.MinLengthValidator(self.min_length, self.errors['min_length']))
        if self.max_length:
            self.validators.append(dj_validators.MaxLengthValidator(self.max_length, self.errors['max_length']))

    def validate_field(self, value: list) -> list:
        """
        Validates list field
        :raise SerializerError:
        """
        try:
            value: list = self.validate_field_raw(value)
        except ValidationError as e:
            raise SerializerError([e])

        if self.unique_items:
            try:
                is_unique = len(value) == len(list(value))
            except TypeError:
                is_unique = False

            if not is_unique:
                raise SerializerError([ValidationError(self.errors['unique_items'], code='unique_items')])

        errors = {}
        converted = []

        if self.item_field:
            for index, item in enumerate(value):
                try:
                    converted.append(self.item_field.validate_field(item))
                except SerializerError as e:
                    errors[str(index)] = e

            if errors:
                raise SerializerError(errors)
        else:
            converted = value

        errors = []
        for validator in self.validators:
            try:
                validator(value)
            except ValidationError as e:
                errors.append(e)
                if settings.ROD_FIRST_ONLY:
                    break

        if errors:
            raise SerializerError(errors)

        return converted


class _TimesField(BaseField):
    default_errors = {
        **BaseField.default_errors,
        'min_value': MinValueValidator.message,
        'max_value': MaxValueValidator.message,
    }
    _parse_func = None

    def __init__(self, *, min_value=None, max_value=None, default=None, **kwargs):
        super().__init__(default=default, **kwargs)
        if min_value:
            self.validators.append(MinValueValidator(min_value, self.errors['min_value']))
        if max_value:
            self.validators.append(MaxValueValidator(max_value, self.errors['max_value']))

    def to_python(self, json_value: str):
        try:
            item = self._parse_func(json_value)
            if item is None:
                raise ValueError()
            return item
        except ValueError:
            raise ValidationError(self.errors['invalid'], 'invalid')


class DateField(_TimesField):
    default_errors = {
        **_TimesField.default_errors,
        'invalid': _('Invalid date.'),
        'min_value': MinValueValidator.message,
        'max_value': MaxValueValidator.message,
    }
    _parse_func = parse_date

    def __init__(
            self, *, min_value: T_VALUE[datetime.date] = None, max_value: T_VALUE[datetime.date] = None,
            default: T_DEFAULT[datetime.date] = None, **kwargs
    ):
        super().__init__(min_value=min_value, max_value=max_value, default=default, **kwargs)

    def to_python(self, json_value: str) -> datetime.date:
        return super().to_python(json_value)

    def to_json(self, python_value: datetime.date) -> str:
        return python_value.isoformat()


class TimeField(_TimesField):
    default_errors = {
        **_TimesField.default_errors,
        'invalid': _('Invalid time.'),
        'min_value': MinValueValidator.message,
        'max_value': MaxValueValidator.message,
    }
    _parse_func = parse_time

    def __init__(
            self, *, min_value: T_VALUE[datetime.date] = None, max_value: T_VALUE[datetime.date] = None,
            default: T_DEFAULT[datetime.date] = None, **kwargs
    ):
        super().__init__(min_value=min_value, max_value=max_value, default=default, **kwargs)

    def to_python(self, json_value: str) -> datetime.time:
        return super().to_python(json_value)

    def to_json(self, python_value: datetime.time) -> str:
        return python_value.isoformat()


class DateTimeField(_TimesField):
    default_errors = {
        **_TimesField.default_errors,
        'invalid': _('Invalid datetime.'),
    }
    _parse_func = parse_datetime

    def __init__(
            self, *, min_value: T_VALUE[datetime.datetime] = None, max_value: T_VALUE[datetime.datetime] = None,
            default: T_DEFAULT[datetime.datetime] = None, **kwargs
    ):
        super().__init__(min_value=min_value, max_value=max_value, default=default, **kwargs)

    def to_python(self, json_value: str) -> datetime.datetime:
        return super().to_python(json_value)

    def to_json(self, python_value: datetime.datetime) -> str:
        return python_value.isoformat()


class DurationField(_TimesField):
    default_errors = {
        **_TimesField.default_errors,
        'invalid': _('Invalid duration.'),
    }
    _parse_func = parse_duration

    def to_python(self, json_value: str) -> datetime.timedelta:
        return super().to_python(json_value)

    def to_json(self, python_value: datetime.timedelta) -> str:
        return timedelta_to_iso8601(python_value)


class SlugField(CharField):
    default_errors = {
        **CharField.default_errors,
        'invalid': _('Invalid slug.'),
    }
    default_slug_regex = re.compile(r'^[a-z0-9]+(-[a-z0-9]+)*$')

    def __init__(self, *, regexes: T_REGEXES = None, **kwargs):
        if regexes is None:
            regexes = (self.default_slug_regex,)
        super().__init__(regexes=regexes, **kwargs)


class URLField(CharField):
    default_errors = {
        **CharField.default_errors,
        'invalid': _('Invalid URL.'),
    }

    def __init__(self, *, schemes: tuple[str] = ('http', 'https', 'ftp', 'ftps'), **kwargs):
        """
        :param schemes: tuple of URL schemes - 'http', 'https', 'ftp', 'ftps'
        """

        super().__init__(**kwargs)
        self.validators.append(URLValidator(schemes=schemes))


class UUIDField(BaseField):
    default_errors = {
        **BaseField.default_errors,
        'invalid': _('Invalid UUID.')
    }

    def __init__(self, *, default: T_DEFAULT[UUID] = _UNDEFINED, **kwargs):
        super().__init__(json_type=str, default=default, **kwargs)

    def to_python(self, json_value: str) -> UUID:
        try:
            uuid = UUID(json_value)
        except ValueError:
            raise ValidationError(self.errors['invalid'], code='invalid')
        return uuid

    def to_json(self, python_value: UUID) -> str:
        return str(python_value)


class EmailField(CharField):
    default_errors = {
        **CharField.default_errors,
        'invalid': _('Invalid email.'),
    }

    def __init__(self, *, domain_allowlist: tuple[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.validators.append(EmailValidator(message=self.errors['invalid'], allowlist=domain_allowlist))


class IPAddressField(BaseField):
    default_errors = {
        **BaseField.default_errors,
        'invalid': _('Invalid IP address.'),
    }

    def __init__(self, *, protocol: str = 'both', **kwargs):
        """
        :param protocol: protocol of an IP address, either 'ipv4', 'ipv6' or 'both'
        """

        super().__init__(json_type=str, **kwargs)

        self.protocol = protocol.lower()
        assert self.protocol in ('both', 'ipv4', 'ipv6'), "Valid values for 'protocol' are 'both', 'ipv4' and 'ipv6'"

        self.validators.append({
            'ipv4': ipv4_address_validator,
            'ipv6': ipv6_address_validator,
            'both': ipv46_address_validator
        }[self.protocol](self.errors['invalid']))
