"""Voice input/output tools with Whisper support"""

import os
import tempfile
from pathlib import Path
from typing import Optional


class VoiceTools:
    @staticmethod
    def listen(timeout: int = 5, phrase_limit: int = 10, language: str = "en") -> str:
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source, duration=0.5)
                audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)

            try:
                text = r.recognize_google(audio, language=language)
                return f"You said: {text}"
            except Exception:
                pass

            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio.get_wav_data())
                    tmp_path = tmp.name

                result = VoiceTools._whisper_transcribe_file(tmp_path)
                os.unlink(tmp_path)

                if result and not result.startswith("Error"):
                    return f"You said: {result}"
            except Exception:
                pass

            try:
                text = r.recognize_google(audio)
                return f"You said: {text}"
            except sr.UnknownValueError:
                return "Could not understand audio"
            except sr.RequestError as e:
                return f"Speech recognition error: {e}"

        except ImportError:
            return "Error: Install speech_recognition (pip install SpeechRecognition)"
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def speak(text: str, rate: int = 150, volume: float = 1.0) -> str:
        try:
            import pyttsx3
            engine = pyttsx3.init()

            voices = engine.getProperty('voices')
            if voices:
                engine.setProperty('voice', voices[0].id)

            engine.setProperty('rate', rate)
            engine.setProperty('volume', volume)

            engine.say(text)
            engine.runAndWait()
            return "Speech output played"
        except ImportError:
            return "Error: Install pyttsx3 (pip install pyttsx3)"
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def save_speech(text: str, file_path: str = "", language: str = "en") -> str:
        try:
            from gtts import gTTS
            output = file_path or str(Path(tempfile.gettempdir()) / "agent_speech.mp3")
            tts = gTTS(text=text, lang=language)
            tts.save(output)
            return f"Speech saved to: {output}"
        except ImportError:
            return "Error: Install gTTS (pip install gtts)"
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def whisper_transcribe(file_path: str, language: str = "") -> str:
        p = Path(file_path)
        if not p.exists():
            return f"Error: File not found: {file_path}"

        return VoiceTools._whisper_transcribe_file(str(p), language)

    @staticmethod
    def _whisper_transcribe_file(file_path: str, language: str = "") -> str:
        try:
            import whisper

            model = whisper.load_model("base")

            options = {}
            if language:
                options["language"] = language

            result = model.transcribe(file_path, **options)

            text = result.get("text", "").strip()
            detected_lang = result.get("language", "unknown")

            if text:
                result_text = f"Transcribed ({detected_lang}): {text}"
                if len(text) > 5000:
                    result_text = result_text[:5000] + "... (truncated)"
                return result_text
            return "No speech detected in audio"

        except ImportError:
            return "Error: Install whisper (pip install openai-whisper)"
        except Exception as e:
            return f"Error transcribing: {e}"

    @staticmethod
    def audio_info(file_path: str) -> str:
        p = Path(file_path)
        if not p.exists():
            return f"Error: File not found: {file_path}"

        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(str(p))

            info = [
                f"Audio: {p.name}",
                f"Duration: {len(audio) / 1000:.1f} seconds",
                f"Channels: {audio.channels}",
                f"Sample width: {audio.sample_width} bytes",
                f"Frame rate: {audio.frame_rate} Hz",
                f"File size: {p.stat().st_size:,} bytes",
            ]

            return "\n".join(info)
        except ImportError:
            return f"File: {p.name}\nSize: {p.stat().st_size:,} bytes"
        except Exception as e:
            return f"Error analyzing audio: {e}"

    @staticmethod
    def list_microphones() -> str:
        try:
            import speech_recognition as sr
            mics = sr.Microphone.list_microphone_names()
            if not mics:
                return "No microphones found"
            lines = ["Available microphones:"]
            for i, name in enumerate(mics):
                lines.append(f"  {i}: {name}")
            return "\n".join(lines)
        except ImportError:
            return "Error: Install speech_recognition"
        except Exception as e:
            return f"Error: {e}"
