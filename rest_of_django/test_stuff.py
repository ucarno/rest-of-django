import django
from django.conf import settings

from fields import Serializer, CharField, SerializerError, IntegerField, ListField

settings.configure(
    USE_I18N=True,
    ROD_FORMAT='code_message',
    ROD_ONLY_FIRST=True,
    APPS=['rest_of_django']
)
django.setup()


class Owl(Serializer):
    test222 = CharField(required=False, regexes=[('fdsafdsa', '1?')])
    test2222 = CharField(required=False, regexes=[('fdsafdsa', '2?')])
    kala = IntegerField(required=True, min_value=3, max_value=3)


class Cat(Serializer):
    test = CharField(required=False, regexes=[('fdsafdsa', '1?')])
    pizza = CharField(required=False, regexes=[('fdsafdsa', '2?'), ('fds', 'rew')])
    owl = Owl(required=True)
    l = ListField(item_field=Owl(), required=False, min_length=1)
    # name = ListField()


class Dog(Cat):
    cat = Cat(required=True)


if __name__ == '__main__':
    obj = Dog(data={
        'cat': {
            'test': '1111',
            'test1': 21,
            'owl': {
                'test222': '1111111', 'test2222': 32432, 'kala': 3
            },
            'l': ''
        }
    })

    obj.is_valid(raise_error=True)
