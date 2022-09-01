# Rest of Django

----
_Heavily work in progress!_

## About
Simple, yet complex REST package for Django.

## Current state
* [x] Everything is typed!
* [ ] Base fields
  * _Done: Boolean, Integer, Decimal, Float, Char, List, Dict_
  * _Todo: Integer (preset size limits for convenience?), Date, Time, DateTime, Slug, URL, UUID, Duration, Email, IPAddress (v4, v6)_
* [ ] Automatic QuerySet builder based on required fields for particular request (with selects and prefetches!)
* [ ] Automatic serializer generation from models
* [ ] Generic views
  * [ ] Snippets + field-based GET through `fields` param
  * [ ] Field-based permissions
* [ ] Async support
  * [ ] Views
  * [ ] Database based validations (unique validations)
* [ ] API documentation generator ([OpenAPI -> Redoc](https://github.com/Redocly/redoc))
* [ ] Package documentation
* [ ] Built-in API debugger for development
