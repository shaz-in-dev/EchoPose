@echo off
cd /d "C:\Users\Admin\wifi vision\desktop"
set CSC_IDENTITY_AUTO_DISCOVERY=false
set WIN_CSC_LINK=
set WIN_CSC_KEY_PASSWORD=
npm run build
echo EXIT_CODE=%ERRORLEVEL%
