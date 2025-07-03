@echo off
echo.
echo ========================================
echo   MCP Document Analyzer - Deploy Setup
echo ========================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

echo ✅ Python found
echo.

:: Run the setup script
echo 🚀 Running deployment setup...
echo.
python deploy_setup.py

echo.
echo ========================================
echo Setup completed!
echo.
echo 📖 Next steps:
echo 1. Check the output above for any errors
echo 2. Read RENDER_DEPLOYMENT_GUIDE.md for detailed instructions
echo 3. Go to render.com to deploy your application
echo ========================================
echo.
pause