from django.apps import AppConfig


class RestOfDjangoConfig(AppConfig):
    name = 'rest_of_django'

    def ready(self):
        from django.conf import settings
        settings = settings._wrapped.__dict__
        settings.setdefault('ROD_FORMAT', 'code_message')  # code | message | code_message
        settings.setdefault('ROD_ONLY_FIRST', False)
