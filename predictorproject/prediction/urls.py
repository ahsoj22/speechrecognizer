from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('submit/', views.submit_prompt, name='submit_prompt'),
    path('predict/', views.predict_word, name='predict_word'),
    path('reset/', views.reset_session, name='reset_session'), 
]
