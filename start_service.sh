#!/bin/bash
# Start API
uvicorn api:app --host 0.0.0.0 --port 8000 &
# Start Streamlit
streamlit run app.py --server.port=8501 --server.address=0.0.0.0 &
# Wait for any process to exit
wait -n
# Exit with status of process that exited first
exit $?