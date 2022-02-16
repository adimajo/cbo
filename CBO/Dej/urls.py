"""
urls module
"""
from django.urls import path

from Dej import views

app_name = 'Dej'
urlpatterns = [
    path('', views.IndexView.as_view(), name="home"),
    path("NewPetitDejView/", views.NewPetitDejView.as_view(), name="NewPetitDejView"),
    path('NewPetitDejViewVue/', views.NewPetitDejViewVue.as_view(), name="NewPetitDejViewVue"),

    path('PetitDej/add/', views.PetitDejCreate.as_view(), name='PetitDej-add'),
    path('PetitDej/<int:pk>/', views.PetitDejUpdate.as_view(), name='PetitDej-update'),
    path('PetitDej/<int:pk>/delete/', views.PetitDejDelete.as_view(), name='PetitDej-delete'),

    path('ListUserView/', views.ListUserView.as_view(), name="ListUserView"),
    path('UserDetailView/<str:email>', views.UserDetailView.as_view(), name="UserDetailView"),

]
