@echo off
rem Copies the built jar into the "My MOD - FABRIC - 1.21.1" instance and makes config\restyler.
set "JAR=%~dp0build\libs\restyler-3.0.0.jar"
set "MC=%APPDATA%\.minecraft\versions"

if not exist "%JAR%" (
    echo Build first:  gradlew build
    pause
    exit /b 1
)
for /d %%D in ("%MC%\My MOD * FABRIC - 1.21.1") do (
    copy /Y "%JAR%" "%%D\mods\restyler-3.0.0.jar"
    if not exist "%%D\config\restyler" mkdir "%%D\config\restyler"
    echo.
    echo Copied to: %%D\mods\
    echo The model downloads itself on first start into: %%D\config\restyler\
    if exist "%%D\mods\snowfilter-0.1.0.jar" echo WARNING: remove %%D\mods\snowfilter-0.1.0.jar - both mods restyle the same frame.
    goto :done
)
echo ERROR: Minecraft version folder not found.
pause
exit /b 1

:done
pause
