from flask import Flask, request, render_template, send_file
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os 
UPLOAD_FOLDER = "uploads"   # change this to YOUR folder name


app = Flask(__name__)

# Load model and encoder
model = joblib.load("weather_model.pkl")
le = joblib.load("label_encoder.pkl")

@app.route("/")
def home():
    return render_template("index.html")
###----------logic for manual entering data---------
@app.route("/predict", methods=["POST"])
def predict():
    temp = float(request.form["tempC"])
    humidity = float(request.form["humidity"])
    wind = float(request.form["windspeedKmph"])

    features = np.array([[temp, humidity, wind]])

    prediction = model.predict(features)
    weather = le.inverse_transform(prediction)[0]

    return render_template("index.html", prediction_text=f"Predicted Weather: {weather}")

###---------logic for csv file----------

@app.route("/predict_csv", methods=["POST"])
def predict_csv():

    file = request.files["file"]

    filename = file.filename
    upload_path = UPLOAD_FOLDER + "/" + filename
    file.save(upload_path)

    df = pd.read_csv(upload_path)

    # CLEAN COLUMN NAMES (IMPORTANT)
    df.columns = df.columns.str.strip()

    # ML FEATURES ONLY
    features = df[["tempC", "humidity", "windspeedKmph"]]

    # PREDICTION
    preds = model.predict(features)
    df["Predicted Weather"] = le.inverse_transform(preds)

    # SAVE FULL DATA (includes date if present)
    output_path = "static/predicted_output.csv"
    df.to_csv(output_path, index=False)

    return render_template(
        "index.html",
        download_link=output_path
    )

@app.route("/trends")
def trends():

    df = pd.read_csv("static/predicted_output.csv")

    # ================= SAFE DATE HANDLING =================
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
        df = df.sort_values("date")
    else:
        df["date"] = pd.date_range(start="2024-01-01", periods=len(df), freq="D")

    # Ensure plots folder exists
    os.makedirs("static/plots", exist_ok=True)

    # ================= GRAPH 1: WEATHER DISTRIBUTION =================
    plt.figure()
    df["Predicted Weather"].value_counts().plot(kind="bar", color="skyblue")
    plt.title("Weather Frequency Distribution")
    plt.xlabel("Weather Type")
    plt.ylabel("Count")
    plt.savefig("static/plots/weather_distribution.png")
    plt.close()


    # ================= GRAPH 2: TEMPERATURE TREND =================
    plt.figure()
    plt.plot(df["date"], df["tempC"])
    plt.title("Seasonal Temperature Trend (Jan–Mar)")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.xticks(rotation=45)
    plt.savefig("static/plots/temp_trend.png")
    plt.close()

    # ================= GRAPH 3: HUMIDITY TREND =================
    plt.figure()
    plt.plot(df["date"], df["humidity"])
    plt.title("Humidity Variation Over Time")
    plt.xlabel("Date")
    plt.ylabel("Humidity (%)")
    plt.xticks(rotation=45)
    plt.savefig("static/plots/humidity_trend.png")
    plt.close()



    # ================= GRAPH 7: BOXPLOT =================
    plt.figure()
    sns.boxplot(x="Predicted Weather", y="tempC", data=df)
    plt.title("Temperature vs Weather")
    plt.savefig("static/plots/temp_weather.png")
    plt.close()

    # ================= GRAPH 9: HUMIDITY vs WEATHER =================
    plt.figure(figsize=(8, 5))

    sns.boxplot(x="Predicted Weather", y="humidity", data=df, palette="coolwarm")

    plt.title("Humidity vs Predicted Weather")
    plt.xlabel("Weather Type")
    plt.ylabel("Humidity (%)")
    plt.xticks(rotation=45)

    plt.savefig("static/plots/humidity_weather.png")
    plt.close()


 
    # ================= Weather Transition Over Time =================
    weather_map = {"Cold": 0, "Cloudy": 1, "Sunny": 2, "Hot": 3, "Rainy": 4}

    df["weather_numeric"] = df["Predicted Weather"].map(weather_map)

    plt.figure()
    plt.plot(df["date"], df["weather_numeric"])
    plt.title("Weather Transition Over Time")
    plt.yticks([0,1,2,3,4], ["Cold","Cloudy","Sunny","Hot","Rainy"])
    plt.xticks(rotation=45)
    plt.savefig("static/plots/weather_trend.png")
    plt.close()
  

    # ================= GRAPH 8: PIE CHART =================
    plt.figure()
    df["Predicted Weather"].value_counts().plot(kind="pie", autopct="%1.1f%%")
    plt.ylabel("")
    plt.title("Weather Share")
    plt.savefig("static/plots/pie.png")
    plt.close()

      # ================= SIMPLE INSIGHTS =================
    insights = []

    # Total records
    insights.append(f"Total records analyzed: {len(df)}")

    # Most common weather
    most_common = df["Predicted Weather"].mode()[0]
    insights.append(f"Most common weather: {most_common}")

    # Average temperature
    avg_temp = df["tempC"].mean()
    insights.append(f"Average temperature: {round(avg_temp, 2)}°C")

    # Temperature range
    temp_range = df["tempC"].max() - df["tempC"].min()
    insights.append(f"Temperature range: {temp_range}°C")

    # Rain percentage
    rain_percentage = (df["Predicted Weather"] == "Rainy").mean() * 100
    insights.append(f"Rainy days: {round(rain_percentage, 2)}%")

    # Humidity level
    avg_humidity = df["humidity"].mean()
    if avg_humidity > 70:
        insights.append("Humidity: High (rainy conditions likely)")
    else:
        insights.append("Humidity: Normal")


    return render_template("trends.html", insights=insights)

    
@app.route("/about")
def about():
    return render_template("about.html")




if __name__ == "__main__":
    app.run(debug=True)
