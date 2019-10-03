import pyttsx3
import speech_recognition as sr


class SpeechEngine:

    def __init__(self, lang):

        # object creation
        self.engine = pyttsx3.init()

        # getting details of current voice
        self.voices = self.engine.getProperty('voices')
        if lang == "en":
            self.engine.setProperty('voice', self.voices[1].id)
            # setting up new voice rate
            self.engine.setProperty('rate', 120)
        if lang == "fr":
            self.engine.setProperty('voice', self.voices[0].id)
            # setting up new voice rate
            self.engine.setProperty('rate', 140)

    def text2speech(self, input_txt):

        self.engine.say(input_txt)
        self.engine.runAndWait()
        self.engine.stop()

    def speech2text(self):

        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("say something !")
            audio = r.listen(source)
            print("time over !")


        try:
            print("text : " + r.recognize_google(audio));
        except:
            pass;
