import inspect
from decimal import Decimal
from functools import cache
from types import MappingProxyType
from typing import Pattern, Iterable, TypeVar, Callable, Any, Optional

from django.conf import settings
from django.core import validators as dj_validators
from django.core.exceptions import ValidationError
from django.db.models import Model
from django.utils.translation import gettext_lazy as _

from rest_of_django.exceptions import SerializerError

T = TypeVar('T')
U = TypeVar('U')

T_DEFAULT = Optional[T | Callable[[], T]]

T_REGEX = str | Pattern
T_REGEXES = (
    T_REGEX,                                           # compiled or non-compiled regex
    Iterable[T_REGEX] |                                # iterable of compiled or non-compiled regexes
    Iterable[tuple[T_REGEX, Optional[str]] | T_REGEX]  # iterable of compiled or non-compiled regexes and their errors
)
T_VALIDATORS = Iterable[Callable[[Any], Any]]
T_DUNDER_CALL = Callable[[Optional[T]], None]

# use this instead of 'None' when 'None' is also a valid value
_UNDEFINED = object()

# empty immutable dict
_EMPTY_DICT = MappingProxyType({})


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
            self.json_type = (self.json_type,) if isinstance(self.json_type, type) else tuple(self.json_type)

        self.allow_null = allow_null
        self.required = required
        self.read_only = read_only
        self.write_only = write_only
        self.snippet = snippet

        # default value, can be a function with no mandatory arguments
        self.default: T_DEFAULT[Any] = default

    @staticmethod
    def to_python(json_value):
        """
        Converts JSON value to Python
        :raises: ValidationError: if there is an error on conversion
        """
        return json_value

    @staticmethod
    def to_json(python_value):
        """Converts Python value to JSON"""
        return python_value

    def validate_field_base(self, value):
        """
        Validates raw value from parsed JSON and converts it to Python value for further validation
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
        Validates converted to python value
        :raises: SerializerError: with list of exceptions if value is invalid
        :return: Converted value
        """

        errors = []
        try:
            converted = self.validate_field_base(value)
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

    def __init__(self, *, instance: Model = None, data: dict = None, is_partial: bool = False, **kwargs):
        """
        :param instance: Instance of a Django model
        :param is_partial: Specifies partial validation (e.g. for PATCH request)
        """
        assert not (kwargs and instance), "You can not use a serializer as a field with passed 'instance' to it"
        assert not (kwargs and data is not None), "You can not use a serializer as a field with passed 'data' to it"
        assert not (data and is_partial), "It seems you wanted to partially validate data you passed as 'data' argument. " \
                                          "If so, 'is_partial' must be passed into `.validate` method"

        kwargs.pop('json_type', None)
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

    def validate_field(self, value: dict, is_partial: bool = False) -> dict:
        """Validates serializer as a field"""
        try:
            value = self.validate_field_base(value)
        except ValidationError as e:
            raise SerializerError([e])

        errors = {}
        converted = {}

        for field_name, field in self.get_fields().items():
            field: BaseField

            if field_name in value:
                if field.read_only:
                    errors[field_name] = SerializerError([
                        ValidationError(message=field.errors['read_only'], code='read_only')
                    ])

                try:
                    converted[field_name] = field.validate_field(value[field_name])
                except SerializerError as error:
                    errors[field_name] = error
            else:
                if field.default:
                    converted[field_name] = field.default() if callable(field.default) else field.default
                elif field.required and not is_partial and not field.read_only:
                    errors[field_name] = SerializerError([ValidationError(field.errors['required'])])

        if errors:
            raise SerializerError(errors)

        return converted

    def is_valid(self, *, is_partial: bool = False, raise_error: bool = False) -> bool:
        """Validates a serializer"""
        if self.data is None:
            raise ValueError("Could not validate, because 'data' was not provided")

        try:
            self.validate_field(self.data, is_partial=is_partial)
            return True
        except SerializerError as e:
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
        required_fields = {k for k, v in fields.items() if v.required}
        return required_fields

    @classmethod
    def get_snippet_fields(cls) -> set[str]:
        """Returns a set of snippet field names"""
        fields = cls.get_fields()
        snippet_fields = {k for k, v in fields.items() if v.snippet}
        return snippet_fields


class BooleanField(BaseField):
    def __init__(self, *, default: T_DEFAULT[bool] = _UNDEFINED, **kwargs):
        super().__init__(**kwargs, json_type=bool, default=default)


class CharField(BaseField):
    default_errors = {
        **BaseField.default_errors,
        'min_length': _('Value must be at least %(min_length)s characters long.'),
        'max_length': _('Value can not exceed %(max_length)s characters.'),
        'invalid': _('Invalid value.')
    }

    def __init__(
            self, *, min_length: int = None, max_length: int = None, regexes: T_REGEXES = None,
            default: T_DEFAULT[str] = _UNDEFINED, **kwargs
    ):
        """
        :param min_length: minimum length of a string
        :param max_length: maximum length of a string
        :param regexes: single regular expression or list of regular expressions in form of [re1, re2, ...] or [(re1, error1), (re2, error2), ...]
        :param default:
        :param kwargs:
        """

        super().__init__(**kwargs, json_type=str, default=default)
        self.default: T_DEFAULT[str]

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
        super().__init__(json_type=kwargs.pop('json_type', (int, float)), **kwargs, default=default)
        self.default: T_DEFAULT[int]

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
        super().__init__(**kwargs, json_type=int, default=default)
        self.default: T_DEFAULT[int]


class FloatField(NumericField):
    def __init__(self, *, default: T_DEFAULT[float] = _UNDEFINED, **kwargs):
        super().__init__(**kwargs, json_type=float, default=default)
        self.default: T_DEFAULT[float]


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
        super().__init__(**kwargs, json_type=float, default=default)

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

    @staticmethod
    def to_json(python_value: Decimal) -> float:
        return float(python_value)

    @staticmethod
    def to_python(json_value: float):
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
        super().__init__(**kwargs, json_type=dict, default=default)
        self.default: T_DEFAULT[dict]

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
            value: dict = self.validate_field_base(value)
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
        super().__init__(**kwargs, json_type=list, default=default)
        self.default: T_DEFAULT[list | tuple]

        assert not (unique_items and isinstance(item_field, (ListField, DictField, Serializer))), \
            f"Can not combine 'unique_items' with 'item_field={item_field.__class__.__name__}', " \
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
            value: list = self.validate_field_base(value)
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
