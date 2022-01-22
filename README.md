# Mycroft Speech To Text (stt) plugin to support Mozila DeepSpeech Release 0.9.x

Enables Mozilla DeepSpeech by directly using a local installation via python binding. 

## Installation

* See https://github.com/mozilla/DeepSpeech/releases/tag/v0.9.3 **Binding** to install DeepSpeech. 
* use `mycroft-pip install git+https://github.com/dalgwen/mycroft-tts-plugin-pico2wave.git` to install this plugin.

## Config

It must be configured locally in your config file and should match the following example:
```json
    "stt": {
        "module": "deepspeech_stt_plug",
        "deepspeech_stt_plug": {
            "model": "/home/pi/mycroft-core/deepspeech/kenlm_de.tflite",
            "scorer": "/home/pi/mycroft-core/deepspeech/kenlm_de.scorer",
            "beam_width": 500,
            "lm_alpha": 0.931289039105002,
            "lm_beta": 1.1834137581510284
            "booster_per_words": {
                "emma": 23.8,
                "starte": 1.9
            }
        }
    }
```
* Only mandatory attribut is "model". Ensure, configured model fits to configured language.
* Default value for "beam_with" is 1024.
* "booster_per_words" is a list of words and boost values, 
see https://deepspeech.readthedocs.io/en/latest/Python-API.html#native_client.python.Model.addHotWord
Thereby, you can boost words from *.intent or *.voc that are otherwise not accurately recognized.
