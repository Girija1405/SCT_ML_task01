import numpy as np
import pandas as pd
import pickle
import os  # Importing os to check for file existence
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# Initialize Flask app
app = Flask(__name__)

# Load and preprocess the data
df = pd.read_csv(r'C:\Users\Girija S.H\Desktop\skill craft\Bengaluru_House_Data.csv')
df.drop(['area_type', 'availability', 'society', 'balcony'], axis=1, inplace=True)
df['location'] = df['location'].fillna('Sarjapur Road')
df['size'] = df['size'].fillna(df['size'].mode()[0])
df['bath'] = df['bath'].fillna(df['bath'].median())

def convert_bhk(x):
    try:
        return int(x.split(' ')[0])
    except:
        return None

df['bhk'] = df['size'].apply(convert_bhk)
df.drop(['size'], axis=1, inplace=True)

def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except:
        return None

df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)
df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']
location_stats = df['location'].value_counts(ascending=False)
location_stats_less_than_10 = location_stats[location_stats <= 10]
df['location'] = df['location'].apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
df = df[((df['total_sqft'] / df['bhk']) >= 300)]

def remove_outliers_sqft(df):
    df_output = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        gen_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
        df_output = pd.concat([df_output, gen_df], ignore_index=True)
    return df_output

df = remove_outliers_sqft(df)
df.drop(['price_per_sqft'], axis=1, inplace=True)

# Model setup
x = df.drop(['price'], axis=1)
y = df['price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

ohe = OneHotEncoder(handle_unknown='ignore')
column_trans = make_column_transformer((ohe, ['location']), remainder='passthrough')
scaler = StandardScaler(with_mean=False)
ridge = Ridge()
pipe = make_pipeline(column_trans, scaler, ridge)
pipe.fit(x_train, y_train)

# Save the trained model
model_path = 'RidgeModel.pkl'
if not os.path.isfile(model_path):
    pickle.dump(pipe, open(model_path, 'wb'))

# Load the saved model
model = pickle.load(open(model_path, 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bath = int(request.form['bath'])
    bhk = int(request.form['bhk'])

    # Format input for the model
    input_data = pd.DataFrame([[location, total_sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    prediction = model.predict(input_data)[0]

    return render_template('index.html', prediction_text=f'Predicted House Price: ₹{prediction:.2f}')

if __name__ == '__main__':
    app.run(debug=True)
