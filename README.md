# ONLINE_FRAUD_TRANSACTION_DETECTION_ML
💳 Online Payment Fraud Detection Dashboard
An interactive and visually enhanced Streamlit dashboard for detecting fraudulent online transactions using a pre-trained machine learning model.


🚀 Features
🌙 Dark/Light Mode Toggle

📥 Upload your CSV file with transaction data

🔍 Predict fraudulent transactions using a trained ML model (fraud_model.pkl)

📊 Visual insights with pie charts, bar plots, histograms, and time series graphs

🎉 Dynamic metrics and glowing animations for better fraud interpretation

💾 Download your prediction results as CSV

📁 File Structure
File	Description
app.py	Main Streamlit application script
fraud_model.pkl	Pre-trained ML model for fraud prediction
online fraud detect.ipynb	Jupyter Notebook containing EDA, model training (not used directly in app)

🧠 Model Input Requirements
Your uploaded CSV should include the same features used during model training. Columns like amount, type, and step enable enhanced visualizations. The app automatically excludes isFraud (if present) before prediction.

📸 Dashboard Preview
<p align="center"> <img src="https://user-images.githubusercontent.com/placeholder/fraud-dashboard.png" alt="Dashboard Screenshot" width="700"/> </p>
🛠️ Installation & Running Locally
bash
Copy
Edit
# Clone the repo
git clone https://github.com/your-username/online-fraud-detection.git
cd online-fraud-detection

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
🧾 Sample CSV Format
csv
Copy
Edit
step,type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest
1,PAYMENT,9839.64,170136.0,160296.36,0.0,0.0
1,TRANSFER,1864.28,21249.0,19384.72,0.0,0.0
...
Make sure the columns match what the model expects.

📊 Visualizations Included
Pie chart of Fraud vs Non-Fraud

Bar chart of transaction types with fraud overlay

Histogram of fraudulent transaction amounts

Line chart of fraud over time (step-wise)

📌 Dependencies
streamlit

pandas

numpy

matplotlib

seaborn

pickle

Install with:


🙋‍♂️ Author
Tarun Desetti
🐳 Feel free to connect or suggest improvements!

📃 License
This project is inspired by an article of MIT. All rights reserved
