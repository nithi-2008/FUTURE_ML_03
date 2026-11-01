@echo off
TITLE TicketAI - Support Ticket Classification & Prioritization Dashboard
echo =====================================================================
echo  Launching TicketAI - ML Task 2 (2026) Support Desk Platform
echo  Future Interns Internship Program ^| Developed by nithi-2008
echo =====================================================================
echo.
echo Launching Streamlit dashboard on port 8503...
py -m streamlit run app.py --server.port 8503 --server.headless false
pause
