import pandas as pd
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)
df = pd.read_csv("clean_data.csv")
pipe = pickle.load(open("XgboostModel.pkl","rb"))

@app.route('/')
def index():
    city = sorted(df['City'].unique())
    area = sorted(df["area"].unique())
    fb = sorted(df['Tenant Preferred'].unique())
    furnish = sorted(df['Furnishing Status'].unique())
    at = sorted(df['Area Type'].unique())
    return render_template('index.html', city=city, fb=fb, furnish=furnish, at=at, area=area)


@app.route('/predict', methods=["POST"])
def predict():
    a = request.form.get("bhk")
    b = request.form.get("size")
    c = request.form.get("at")
    d = request.form.get("city")
    e = request.form.get("furnish")
    f = request.form.get("fb")
    g = request.form.get("bath")
    h = request.form.get("floor")
    i = request.form.get("area")
    
    
    print(a,b,c,d,e,f,g,h,i)
    input = pd.DataFrame([[a,b,c,d,e,f,g,h,i]], columns = ["BHK", "Size", "Area Type","City","Furnishing Status","Tenant Preferred","Bathroom", "floor","area"])
    prediction = pipe.predict(input)[0]
    return str(prediction)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
