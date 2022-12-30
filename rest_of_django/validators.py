import ipaddress

from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _


class ChoiceValidator:
    code = 'invalid_choice'
    message = _('Invalid choice. Valid choices are: %(choices)s.')

    def __init__(self, choices, message = None):
        self.choices = set(choices)
        self.message = message or self.message
        self.params = ', '.join(choices)  # save order

    def __call__(self, value: str):
        if value not in self.choices:
            raise ValidationError(message=self.message, code=self.code, params={'choices': self.params})


def ipv4_address_validator(error):
    def validate(value):
        try:
            ipaddress.IPv4Address(value)
        except ValueError:
            raise ValidationError(error, code='invalid')
    return validate


def ipv6_address_validator(error):
    def validate(value):
        try:
            ipaddress.IPv6Address(value)
        except ValueError:
            raise ValidationError(error, code='invalid')
    return validate


def ipv46_address_validator(error):
    def validate(value):
        try:
            ipaddress.IPv4Address(value)
        except ValueError:
            try:
                ipaddress.IPv6Address(value)
            except ValueError:
                raise ValidationError(error, code='invalid')
    return validate
