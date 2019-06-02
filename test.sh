#!/usr/bin/env bash

mypy \
    *.py \
    realtime_voice_conversion \
    tests \
    --ignore-missing-imports \

python -m unittest discover tests
