import wikipedia
import re


class WikiEngine:

    def __init__(self, lang):

        self.lang = lang
        if lang == "fr":
            wikipedia.set_lang("fr")

        if lang == "fr":
            wikipedia.set_lang("fr")

    def run(self, object, sentences_number):

        if self.lang == "en":
            try:
                output = wikipedia.summary(object, sentences=sentences_number)

                # remove hexa chars from encoding problem
                # infos = re.sub(r'[^\x00-\x7f]',r'', output)

                # removing listen key from text
                # infos = infos.replace("(listen)", " ")
                infos = output
            except:
                infos = "Please reask your question in another form !"

        if self.lang == "fr":
            try:
                output = wikipedia.summary(object, sentences=sentences_number)

                # remove hexa chars from encoding problem
                # infos = re.sub(r'[^\x00-\x7f]', r'', output)

                # removing listen key from text
                # infos = infos.replace("(écouter)", " ")
                infos = output
            except:
                infos = "s'il vous plait reformulez votre question !"

        return infos