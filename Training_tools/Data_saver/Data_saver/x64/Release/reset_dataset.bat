@ECHO OFF

:choice
set /P c=Ez ki fogja torolni az egesz datasetet (Szines kepek, melyseg kepek, melyseg adatok es ROIk). Biztos torolsz mindent[Y/N]?
if /I "%c%" EQU "Y" goto :delete
if /I "%c%" EQU "N" goto :cancel
goto :choice


:delete

del /Q master.txt
echo 0 >> master.txt
cd %~dp0color
del /Q *.*
cd %~dp0depth_color
del /Q *.*
cd %~dp0depth_data
del /Q *.*
cd %~dp0ROIs
del /Q *.*
copy NUL ROIs.txt

:cancel