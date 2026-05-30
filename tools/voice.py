"""Voice input/output tools"""

import os
import tempfile
from pathlib import Path
from typing import Optional


class VoiceTools:
    @staticmethod
    def listen(timeout: int = 5, phrase_limit: int = 10) -> str:
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source, duration=0.5)
                audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)
            text = r.recognize_google(audio)
            return f"You said: {text}"
        except ImportError:
            return "Error: Install speech_recognition (pip install SpeechRecognition)"
        except sr.WaitTimeoutError:
            return "No speech detected"
        except sr.UnknownValueError:
            return "Could not understand audio"
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def speak(text: str) -> str:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            return "Speech output played"
        except ImportError:
            return "Error: Install pyttsx3 (pip install pyttsx3)"
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def save_speech(text: str, file_path: str = "") -> str:
        try:
            from gtts import gTTS
            output = file_path or str(Path(tempfile.gettempdir()) / "agent_speech.mp3")
            tts = gTTS(text=text, lang="en")
            tts.save(output)
            return f"Speech saved to: {output}"
        except ImportError:
            return "Error: Install gTTS (pip install gtts)"
        except Exception as e:
            return f"Error: {e}"
