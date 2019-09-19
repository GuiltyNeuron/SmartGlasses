import wikipedia


class WikiEngine:

    def __init__(self):
        wikipedia.set_lang("en")

    def run(self, object, sentences_number):

        infos = wikipedia.summary(object, sentences=sentences_number)
        return infos

we = WikiEngine()

infos = we.run("obama", 3)

print(infos)