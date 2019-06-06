from setuptools import setup, find_packages

setup(
    name='realtime-voice-conversion',
    version='0.1.0',
    packages=find_packages(),
    url='https://github.com/Hiroshiba/realtime-voice-conversion',
    author='Kazuyuki Hiroshiba',
    author_email='hihokaruta@gmail.com',
    description='Realtime Voice Conversion Library With DeepLearning Power.',
    license='MIT License',
    install_requires=[
        'yukarin',
        'become-yukarin',
    ],
    dependency_links=[
        'https://github.com/Hiroshiba/yukarin/master',
        'https://github.com/Hiroshiba/become-yukarin/master',
    ],
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: MIT License',
    ]
)
