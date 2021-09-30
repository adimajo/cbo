REM #================================= Initiate variables ==============================
set port=9000

REM Desired browser
set path_app=C:\"Program Files (x86)\Mozilla Firefox"\firefox.exe
REM set path_app=C:\"Program Files\internet explorer"\iexplore.exe
REM set path_app=C:\Users\DAMAYNI\AppData\Local\Google\Chrome\Application\chrome.exe


REM #================================= Start Browser==============================
START %path_app% "http://127.0.0.1:%port%"

REM #================================= Start Django ==============================
REM # Should not be needed...
REM python manage.py makemigrations
REM python manage.py migrate
REM # Clear Js, Css, ...
REM python manage.py collectstatic --noinput --clear
REM START python manage.py runserver %port%
START python manage.py runserver --noreload %port%





