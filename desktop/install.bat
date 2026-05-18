@echo off
cd /d "C:\Users\Admin\wifi vision\desktop"
if exist node_modules (
    del /f /q node_modules\electron\dist\resources\default_app.asar 2>nul
    rmdir /s /q node_modules\electron\dist\resources 2>nul
    rmdir /s /q node_modules 2>nul
)
npm install
echo EXIT_CODE=%ERRORLEVEL%
