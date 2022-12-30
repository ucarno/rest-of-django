from datetime import timedelta


def timedelta_to_iso8601(value: timedelta):
    """Converts a `datetime.timedelta` to an ISO 8601 duration string"""
    # https://github.com/RusticiSoftware/TinCanPython/blob/3.x/tincan/conversions/iso8601.py

    # Copyright 2014 Rustici Software
    #
    #    Licensed under the Apache License, Version 2.0 (the "License");
    #    you may not use this file except in compliance with the License.
    #    You may obtain a copy of the License at
    #
    #    http://www.apache.org/licenses/LICENSE-2.0
    #
    #    Unless required by applicable law or agreed to in writing, software
    #    distributed under the License is distributed on an "AS IS" BASIS,
    #    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    #    See the License for the specific language governing permissions and
    #    limitations under the License.

    assert isinstance(value, timedelta)

    # split seconds to larger units
    seconds = value.total_seconds()
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)

    days, hours, minutes = list(map(int, (days, hours, minutes)))
    seconds = round(seconds, 6)

    # build date
    date = ''
    if days:
        date = '%sD' % days

    # build time
    time = 'T'

    # hours
    bigger_exists = date or hours
    if bigger_exists:
        time += '{:02}H'.format(hours)

    # minutes
    bigger_exists = bigger_exists or minutes
    if bigger_exists:
        time += '{:02}M'.format(minutes)

    # seconds
    if seconds.is_integer():
        seconds = '{:02}'.format(int(seconds))
    else:
        # 9 chars long w/leading 0, 6 digits after decimal
        seconds = '%09.6f' % seconds
        # remove trailing zeros
        seconds = seconds.rstrip('0')

    time += '{}S'.format(seconds)

    return 'P' + date + time
