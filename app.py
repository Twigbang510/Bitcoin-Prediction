# Import necessary libraries
from flask import Flask, render_template, request, make_response
from datetime import datetime, timedelta
from keras.models import load_model
import numpy as np  
from sklearn.preprocessing import MinMaxScaler  
import yfinance as yf  
import matplotlib.pyplot as plt  
import io
import base64
import pandas as pd
import matplotlib
matplotlib.use('Agg')

# Load the trained model
model_1 = load_model('model_1.h5')

# Initialize Flask app
app = Flask(__name__)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling form submission and displaying predictions
@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        input_date = request.form['input_date']

        # Load historical data for model input (adjust start_date and end_date)
        start_date = '2023-01-01'
        end_date = '2023-11-22'
        df = yf.download('BTC-USD', start=start_date, end=end_date)

        # Prepare data for prediction
        input_date = datetime.strptime(input_date, '%Y-%m-%d')
        future_dates = [input_date + timedelta(days=i) for i in range(60)]
        last_60_days = df['Close'].values[-60:]
        last_60_days_scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(last_60_days.reshape(-1, 1))
        new_X_test = np.array([last_60_days_scaled])

        # Perform prediction using model_1
        pred_price_1 = model_1.predict(new_X_test)
        pred_price_1 = MinMaxScaler(feature_range=(0, 1)).fit(df['Close'].values.reshape(-1, 1)).inverse_transform(pred_price_1)

        # Ensure the lengths of 'Date' and 'Predicted Price' are the same
        future_dates = future_dates[:len(pred_price_1)]

        # Prepare data for plotting
        plot_data_1 = {
            'Date': [date.strftime('%Y-%m-%d') for date in future_dates],
            'Predicted Price': pred_price_1.flatten()
        }

        # Convert data to pandas DataFrame
        df_plot = pd.DataFrame(plot_data_1)

        # Generate plot
        plt.figure(figsize=(10, 6))
        plt.plot(df_plot['Date'], df_plot['Predicted Price'], label='Predicted Price', marker='o')
        plt.title('Bitcoin Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Predicted Price (USD)')
        plt.xticks(rotation=45)
        plt.legend()

        # Save the plot to a BytesIO object
        img_data = io.BytesIO()
        plt.savefig(img_data, format='png')
        img_data.seek(0)

        # Close plot to release resources
        plt.close()

        # Convert plot to base64 for embedding in HTML
        encoded_img = base64.b64encode(img_data.read()).decode('utf-8')

        return render_template('index.html', plot_image=encoded_img)

    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)
