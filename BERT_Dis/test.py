import unittest
from app import app
from google.cloud import speech

class SpeechToTextTestCase(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.speech_client = speech.SpeechClient()

    def test_speech_to_text(self):
        with open('test_audio.wav', 'rb') as audio_file:
            audio_content = audio_file.read()
            audio = speech.RecognitionAudio(content=audio_content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="en-US"
            )
            response = self.speech_client.recognize(config=config, audio=audio)
            self.assertIsNotNone(response)

if __name__ == '__main__':
    unittest.main()