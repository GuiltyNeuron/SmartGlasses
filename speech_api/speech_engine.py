import pyttsx3


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

"""import pyttsx3

# object creation
engine = pyttsx3.init()

# RATE
# getting details of current speaking rate
rate = engine.getProperty('rate')

# setting up new voice rate
engine.setProperty('rate', 160)


# VOLUME
# getting to know current volume level (min=0 and max=1)
volume = engine.getProperty('volume')

# setting up volume level  between 0 and 1
engine.setProperty('volume',1.0)

# VOICE
# getting details of current voice
voices = engine.getProperty('voices')
# changing index, changes voices. 1 for female

engine.setProperty('voice', voices[1].id)

engine.say("Hello World!")

engine.runAndWait()"""

# ================================================================================================
"""from gtts import gTTS
import os

tts = gTTS(text='Good morning', lang='en')
# tts.save("good.mp3")
# os.system("mpg321 good.mp3")

import speech
speech.say('Hola mundo', 'es_ES')

import sound

r = sound.Recorder('audio.m4a')
r.record(3)  # seconds

text = speech.recognize('audio.m4a', 'en')[0][0]  # sent to Apple servers
"""

# ==================================================================================================

"""import win32com.client as wincl
speak = wincl.Dispatch("SAPI.SpVoice")
speak.Speak("Hello World")"""

# ==================================================================================================

"""from tts_watson.TtsWatson import TtsWatson

ttsWatson = TtsWatson('watson_user', 'watson_password', 'en-US_AllisonVoice')
ttsWatson.play("Hello World")"""