@echo off
echo ========================================================
echo  Medical Insurance Cost Predictor - Local Test Runner
echo ========================================================
echo.
echo [1/2] Verifying and training ML models...
python src/train_model.py
echo.
echo [2/2] Launching Streamlit application...
streamlit run app.py
pause
