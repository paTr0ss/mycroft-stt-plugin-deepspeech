#!/usr/bin/env python3
from os.path import join, basename, abspath, dirname
from setuptools import setup

with open(join(dirname(abspath(__file__)), 'requirements.txt')) as f:
    requirements = f.readlines()

PLUGIN_ENTRY_POINT = ('deepspeech_stt_plug = '
                      'mycroft_stt_plugin_deepspeech:DeepspeechSTTPlugin')

setup(
    name='mycroft-stt-plugin-deepspeech',
    version='0.3',
    description='A STT plugin for mycroft using mozilla deepspeech',
    url='https://github.com/forslund/mycroft-stt-plugin-deepspeech',
    author='Joshua Watts',
    author_email='',
    license='Apache-2.0',
    packages=['mycroft_stt_plugin_deepspeech'],
    install_requires=requirements,
    zip_safe=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: STT',
        'License :: OSI Approved :: Apache Software License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='mycroft plugin stt',
    entry_points={'mycroft.plugin.stt': PLUGIN_ENTRY_POINT}
)
