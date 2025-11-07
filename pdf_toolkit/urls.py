from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('server_info/', views.server_info, name='server_info'),  # optional to have it here too
]