import signal

import time

import pynput


def handler(*args, **kwargs):
    print('SIGINT')


def key_handler(key):
    print('key', key)


signal.signal(signal.SIGINT, handler)

key_listener = pynput.keyboard.Listener(
    on_press=key_handler,
)
key_listener.start()

for _ in range(5):
    time.sleep(1)
    print('sleep')
