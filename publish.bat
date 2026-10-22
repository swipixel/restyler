@echo off
rem 1) uploads hf\ (model, styles.json, model card) to https://huggingface.co/SwiPixel/RestylerV3
rem 2) makes this folder a git repo with one commit (pushing to GitHub is up to you, see the end)
setlocal
cd /d "%~dp0"
set REPO=SwiPixel/RestylerV3

echo === Hugging Face ===
where hf >nul 2>nul || (
    echo Installing the Hugging Face CLI...
    python -m pip install -U "huggingface_hub[cli]" || goto :fail
)
where hf >nul 2>nul || (echo "hf" is not on PATH yet - open a new terminal and run publish.bat again. & goto :fail)
hf auth whoami || (
    echo Not logged in. Run:  hf auth login   and paste a WRITE token from huggingface.co/settings/tokens
    goto :fail
)
rem creates the repo (public) if it doesn't exist yet
hf upload %REPO% hf . --commit-message "Restyler V3: ep60 + partial model + styles.json" || goto :fail
echo Uploaded. Check: https://huggingface.co/%REPO%/tree/main

echo.
echo === git ===
where git >nul 2>nul || (echo git not found - skipped. & goto :done)
if not exist .git git init -b main
git add -A
git commit -m "Restyler V3 (3.0.0)" || echo (nothing new to commit, or set: git config --global user.name / user.email)
echo.
echo To put it on GitHub:   gh repo create SwiPixel/restyler --public --source . --push
echo   (or create an empty repo on github.com, then: git remote add origin URL ^& git push -u origin main)

:done
echo.
echo Done.
pause
exit /b 0

:fail
echo.
echo FAILED - see the message above.
pause
exit /b 1
