from django.urls import path  # Django URL dispatcher helper

from . import views  # Local view module

urlpatterns = [
    path('', views.home, name='home'),
    path('server_info/', views.server_info, name='server_info'),  # optional to have it here too
    path('extract/', views.extract_view, name='extract'),
    path('merge-create/', views.merge_create_view, name='merge_create'),
    path('merge-create/<str:record_id>/download/', views.download_merge_result, name='merge_download'),
    path('split/', views.splitter_view, name='splitter'),
    path('split/<str:record_id>/<str:segment_id>/download/', views.download_split_result, name='split_download'),
]
