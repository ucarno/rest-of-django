import django
from django.conf import settings

from fields import Serializer, CharField, SerializerError, IntegerField, ListField


settings.configure(USE_I18N=True)  # todo: error format settings
django.setup()


class Owl(Serializer):
    test222 = CharField(is_required=False, regexes=[('fdsafdsa', '1?')])
    test2222 = CharField(is_required=False, regexes=[('fdsafdsa', '2?')])
    kala = IntegerField(is_required=True, min_value=3, max_value=3)


class Cat(Serializer):
    test = CharField(is_required=False, regexes=[('fdsafdsa', '1?')])
    pizza = CharField(is_required=False, regexes=[('fdsafdsa', '2?')])
    owl = Owl(is_required=False)
    l = ListField(item_field=Owl(), is_required=False, min_length=1)
    # name = ListField()


class Dog(Cat):
    cat = Cat(is_required=True)


if __name__ == '__main__':
    pass
    # try:
    #     Dog(data={'cat': {'test': '1111', 'test1': 21, 'owl': {'test222': '1111111', 'test2222': 32432, 'kala': 3}, 'l': ''}}).is_valid(raise_error=True)
    # except SerializerError as e:
    #     print(e.parse_data(only_messages=False, only_first=False))
