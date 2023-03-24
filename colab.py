from IPython.display import Javascript
from IPython.display import Audio
from IPython.display import display
from google.colab import output
from base64 import b64decode
import openai
import sys
from boto3 import client

openai.api_key = 'your_openai_api_key'

RECORD = """
  const sleep = time => new Promise(resolve => setTimeout(resolve, time))
  const b2text = blob => new Promise(resolve => {
    const reader = new FileReader()
    reader.onloadend = e => resolve(e.srcElement.result)
    reader.readAsDataURL(blob)
  })
  var record = time => new Promise(async resolve => {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    recorder = new MediaRecorder(stream)
    chunks = []
    recorder.ondataavailable = e => chunks.push(e.data)
    recorder.start()
    await sleep(time)
    recorder.onstop = async ()=>{
      blob = new Blob(chunks)
      text = await b2text(blob)
      resolve(text)
    }
    recorder.stop()
  })
"""

def speech_to_text(model='whisper-1', language='en', second=3):
  filename='tmp.wav'
  display(Javascript(RECORD))
  s = output.eval_js('record(%d)' % (second * 1000))
  b = b64decode(s.split(',')[1])

  with open(filename, 'wb+') as fw:
    fw.write(b)

  with open(filename, "rb") as fr:
    transcription = openai.Audio.transcribe(
        model=model, 
        file=fr, 
        language=language
    )
    return transcription['text']
  
def test_to_speech(text):
  polly = client("polly", region_name="us-east-1", aws_access_key_id="YOUR_ACCESS_KEY_ID", aws_secret_access_key="YOUR_SECRET_ACCESS_KEY")
  response = polly.synthesize_speech(
        Text = text,
        OutputFormat = "mp3",
        VoiceId = "Ruth",
        Engine = "neural")

  file = open("test.mp3", "wb")
  file.write(response["AudioStream"].read())
  file.close()

  wn = Audio("test.mp3", autoplay=True)
  display(wn)
  
# 初期のプロンプト
messages = [
  # You are a helpful assistant.
  # You are a talkative comedian.
  # A person who enjoys short conversations.
  {"role": "system", "content": "A person who enjoys short conversations."},
  {"role": "user", "content": "Hello"},
]

while True:
  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", 
    messages=messages
  )

  bot_speech = completion.choices[0].message.content
  test_to_speech(bot_speech)
  for i in range(0, len(bot_speech), 100):
    print(bot_speech[i : i + 100])

  while True:
    input()
    user_speech = speech_to_text()
    if user_speech != '':
      break
  print(f"user: {user_speech}")

  messages.extend([
      {"role": "assistant", "content": bot_speech},
      {"role": "user", "content": user_speech}
  ])
  
  if user_speech == 'end' or user_speech == 'End' or user_speech == 'End.':
    break