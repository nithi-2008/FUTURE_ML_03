@echo off
TITLE TalentMatch AI - Resume & Candidate Screening Dashboard
echo =====================================================================
echo  Launching TalentMatch AI - Machine Learning Task 3 (2026)
echo  Resume / Candidate Screening & Decision-Support System
echo  Future Interns Internship Program ^| Developed by nithi-2008
echo  Repository: https://github.com/nithi-2008/FUTURE_ML_03.git
echo =====================================================================
echo.
echo Launching Streamlit web dashboard on port 8505...
py -m streamlit run app.py --server.port 8505 --server.headless false
pause
