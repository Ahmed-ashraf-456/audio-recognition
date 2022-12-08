from flask import Flask

app= Flask(__name__)

from voice_recognition import routes , index
