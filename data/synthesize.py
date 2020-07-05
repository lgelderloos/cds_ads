from google.cloud import texttospeech
import random
import json
import time

def clean_sentence(sentence):
    clean = []
    for word in sentence:
        word = word.lower()
        # clean out underscores in fixed expressions
        if "_" in word:
            clean.extend(word.split("_"))
        else:
            clean.append(word)
    # return as strng rather than as wordlist
    return " ".join(clean)

def speak(sentence, client, encoding, speaker="random"):
    # make a clean string without nderscores
    text = clean_sentence(sentence)
    # select voice (use wavenet voices as they are so muh better)
    if speaker == "random":
        # these are all en-US WaveNet voices available in the API (Nov 5 2019)
        # A, B and D are 'male', 'C', 'E' and 'F' are 'female'
        voices = ["en-US-Wavenet-A", "en-US-Wavenet-B", "en-US-Wavenet-C",
                  "en-US-Wavenet-D", "en-US-Wavenet-E", "en-US-Wavenet-F"]
        speaker = random.choice(voices)
    voice = texttospeech.types.VoiceSelectionParams(name=speaker, language_code='en-US')
    # Select the type of audio file you want returned - using LINEAR16 for now
    audio_config = texttospeech.types.AudioConfig(audio_encoding=encoding)
    # Set the text input to be synthesized
    synthesis_input = texttospeech.types.SynthesisInput(text=text)
    # Perform the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
    try:
        spoken = client.synthesize_speech(synthesis_input, voice, audio_config)
        return spoken
    except:
        # if you run out of quotum, sleep for a minute
        time.sleep(60)
        # try again
        try:
            spoken = client.synthesize_speech(synthesis_input, voice, audio_config)
            return spoken
        # if still fails aftr sleeping, skip, report to user
        except:
            return False

# Instantiate a google tts client
client = texttospeech.TextToSpeechClient()
encoding = texttospeech.enums.AudioEncoding.LINEAR16
# keep track of failed wavs
failed = {"ADS" : [], "CDS" : []}

# open ADS json
# speak every sentence and store the speech as a wav
ADS = json.load(open('ADS.json'))
for utterance in ADS:
    wav = str(utterance) + '.wav'
    response = speak(ADS[utterance]['sentence'], client, encoding)
    if response:
        with open('synthetic_speech/ADS/{}'.format(wav), 'wb') as out:
            out.write(response.audio_content)
    else:
        failed["ADS"].append(utterance)
        print("Failed to process ADS {}".format(utterance), flush=True)

# open CDS json
CDS = json.load(open('CDS.json'))
for utterance in CDS:
    wav = str(utterance) + '.wav'
    response = speak(CDS[utterance]['sentence'], client, encoding)
    if response:
        with open('synthetic_speech/CDS/{}'.format(wav), 'wb') as out:
            out.write(response.audio_content)
    else:
        print("Failed to process CDS {}".format(utterance), flush=True)
        failed["CDS"].append(utterance)

with open("failed.json", "w") as f:
    json.dump(failed, f)
