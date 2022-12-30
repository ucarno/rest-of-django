# Rest of Django
_Heavily work in progress!_

## About
Simple, yet complex REST package for Django.

## Current state
* [x] Everything is typed!
* [x] Base fields
* [ ] Automatic QuerySet builder based on required fields for particular request (with selects and prefetches!)
* [ ] Automatic serializer generation from models
* [ ] Generic views
  * [ ] Snippets + field-based GET through `fields` param
  * [ ] Field and request method based permissions
* [ ] Async support
  * [ ] Views
  * [ ] Validations (e.g. database uniqueness validations, custom user-defined validations)
* [ ] API documentation generator ([OpenAPI -> Redoc | Swagger UI](https://github.com/Redocly/redoc))
* [ ] Package documentation
* [ ] Built-in API debugger for development
