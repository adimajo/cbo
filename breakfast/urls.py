from django.urls import path

from breakfast import views

app_name = "breakfast"
urlpatterns = [
    path("new/", views.NewBreakfastView.as_view(), name="new"),
    path("api/drafts/", views.DraftCreateView.as_view(), name="draft-create"),
    path("api/drafts/<str:token>/", views.DraftUpdateView.as_view(), name="draft-update"),
    path("users/", views.UserListView.as_view(), name="users"),
    path("users/<int:pk>/", views.UserDetailView.as_view(), name="user-detail"),
]
