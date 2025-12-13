from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required #requires login
from django.shortcuts import redirect, render

from .forms import JoinForm, LoginForm


def signup(request):
    if request.method == "POST":
        form = JoinForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect("home")
    else:
        form = JoinForm()
    return render(request, "registration/signup.html", {"form": form})


def user_login(request):
    if request.method == "POST":
        #authentication fporm performs val
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect("home")
    else:
        form = LoginForm(request)
    return render(request, "registration/login.html", {"form": form})


@login_required
def user_logout(request):
    logout(request)
    return redirect("home")
