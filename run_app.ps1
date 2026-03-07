# Star AI - Streamlit Web App Launcher (PowerShell)

Write-Host "========================================"
Write-Host " Star AI - Constellation Detection UI  " -ForegroundColor Cyan
Write-Host "========================================"
Write-Host ""
Write-Host "Starting Streamlit server..." -ForegroundColor Yellow
Write-Host ""

# Check if streamlit is installed
try {
    $streamlitCheck = python -m streamlit --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Streamlit not found. Installing..." -ForegroundColor Yellow
        pip install streamlit plotly
    }
} catch {
    Write-Host "Installing Streamlit..." -ForegroundColor Yellow
    pip install streamlit plotly
}

# Run the app
streamlit run app.py
