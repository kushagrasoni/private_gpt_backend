# Start Frontend
cd private_gpt_frontend/
npm start


# Start Backend
source venv311_pvt_gpt/bin/activate
cd private_gpt_backend/
python -m app.main

# Execute backend model
curl -v http://localhost:5000
curl -v http://localhost:5000/run_model