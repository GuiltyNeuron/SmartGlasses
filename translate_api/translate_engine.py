from translate import Translator
from nltk.tokenize import sent_tokenize
import unicodedata


class TranslateEngine():

    def __init__(self, language):

        # Translator class instance
        self.translator = Translator(to_lang=language)

    def strip_accents(self, text):
        return "".join(char for char in
                       unicodedata.normalize('NFKD', text)
                       if unicodedata.category(char) != 'Mn')

    def translate(self, text):

        # Output translated text variable
        translated = ""

        # Subdevise text into sentences because the translate can translate only 500 char per step
        sentences = sent_tokenize(text)

        # Loop over sentences
        for s in sentences:

            # Translate sentence
            translation = self.translator.translate(s)

            # Append all sentences to get all the text
            translated = translated + " " + translation

        # Remove accents, in french lang
        output = self.strip_accents(translated)
        return output
