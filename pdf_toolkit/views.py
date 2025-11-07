# Create your views here.
# Functions that handle upload, login, etc.

from django.http import HttpResponse
import requests
from django.conf import settings

def home(request):
    return HttpResponse("BIG TESTIE")

def server_info(request):
    server_geodata = requests.get('https://ipwhois.app/json/').json()
    settings_dump = settings.__dict__
    return HttpResponse(f"{server_geodata}{settings_dump}")