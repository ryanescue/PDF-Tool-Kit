from django.urls import path  # Django URL dispatcher helper

from . import views  # Local view module

urlpatterns = [
    path('', views.home, name='home'),
    path('server_info/', views.server_info, name='server_info'),  # optional to have it here too
    path('extract/', views.extract_view, name='extract'),
    path('extract/<int:artifact_id>/download/', views.download_extract_result, name='extract_download'),
    path('extract/<int:artifact_id>/preview/', views.extract_preview, name='extract_preview'),
    path('merge-create/', views.merge_create_view, name='merge_create'),
    path('merge-create/<int:artifact_id>/download/', views.download_merge_result, name='merge_download'),
    path('split/', views.splitter_view, name='splitter'),
    path('split/<int:artifact_id>/download/', views.download_split_result, name='split_download'),
]
