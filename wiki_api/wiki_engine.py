import wikipedia
import re


class WikiEngine:

    def __init__(self):
        wikipedia.set_lang("en")

    def run(self, object, sentences_number):

        output = wikipedia.summary(object, sentences=sentences_number)

        # remove hexa chars from encoding problem
        infos = re.sub(r'[^\x00-\x7f]',r'', output)

        # removing listen key from text
        infos = infos.replace("(listen)", " ")
        return infos