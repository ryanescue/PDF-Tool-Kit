from django.urls import path  # Django URL dispatcher helper

from . import views  # Local view module

urlpatterns = [
    path('', views.home, name='home'),
    path('server_info/', views.server_info, name='server_info'),  # optional to have it here too
    path('extract/', views.extract_view, name='extract'),
]
