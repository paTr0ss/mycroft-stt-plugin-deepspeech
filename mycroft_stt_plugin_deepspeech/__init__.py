import deepspeech
import numpy
from mycroft.stt import STT
from mycroft.util.log import LOG


class DeepspeechSTTPlugin(STT):
    """
        Mozilla Deepspeech Speech to Text
        Enables Mozilla Deepspeech by directly using a local installation via python binding. 
        Deepspeech must be installed. It must be configured locally in your config file and
        should match the following example:

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

        Only mandatory attribut is "model". Ensure, configured model fits to configured language.

        Default value for "beam_with" is 1024.

        "booster_per_words" is a list of words and boost values, 
        see https://deepspeech.readthedocs.io/en/latest/Python-API.html#native_client.python.Model.addHotWord
        Thereby, you can boost words from *.intent or *.voc that are otherwise not accurately recognized.
        
    """

    def __init__(self):
        super().__init__()

        model = self.config.get('model', None)
        if not model:
            raise Exception('model not configured!')

        LOG.info(f"Loading DeepSpeech model {model}.")
        self.ds = deepspeech.Model(model)

        beam_width = self.config.get('beam_width', 1024)
        LOG.info(f"Set beam width {beam_width}.")
        self.ds.setBeamWidth(beam_width)

        scorer = self.config.get('scorer', None)
        if scorer:
            LOG.info(f"Enabling scorer {scorer}.")
            self.ds.enableExternalScorer(scorer)

            lm_alpha = self.config.get('lm_alpha', None)
            lm_beta = self.config.get('lm_beta', None)
            if lm_alpha and lm_beta:
                LOG.info(f"Set alpha {lm_alpha} and beta {lm_beta}.")
                self.ds.setScorerAlphaBeta(lm_alpha, lm_beta)

        boosters_per_word = self.config.get('booster_per_words', None)
        if boosters_per_word:
            for w in boosters_per_word:
                LOG.info(f"Add boost {boosters_per_word[w]} for {w}.")
                self.ds.addHotWord(w, boosters_per_word[w])


    def execute(self, audio, language=None):
        if language and language != self.lang:
            LOG.warning(f"Given language {language} ignored, using {self.lang}")

        # LOG.info(f"sample rate: {self.ds.sampleRate:6.2f} == {audio.sample_width:6.2f}")
        LOG.info("Perform stt audio ...")
        text = self.ds.stt(numpy.frombuffer(audio.get_raw_data(), numpy.int16))

        LOG.info(text)
        return text
