from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
import traceback

app = Flask(__name__)

# --- Load and process dataset ---
data = pd.read_csv("abalone.csv").dropna()

X = data[['Sex', 'Length', 'Diameter', 'Height', 'Whole weight']].copy()
y_rings = data['Rings']

le_sex = LabelEncoder()
X['Sex'] = le_sex.fit_transform(X['Sex'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model_rings = LinearRegression()
model_rings.fit(X_scaled, y_rings)

data['WaterType'] = np.where(
    (data['Rings'] > 10) | (data['Whole weight'] > 1.0),
    'Marine',
    'Freshwater'
)

# -----------------------------------
# HTML TEMPLATE (GRAPH REMOVED)
# -----------------------------------
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Abalone Age Predictor</title>
    <style>
        html, body {
            background-color: #f5f5f5;
            margin: 0;
            padding: 0;
        }
        h1 { text-align: center; }

        body.bg-image {
            background-image: url('{{ url_for("static", filename="pearl.jpg") }}');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            color: white;
        }

        .container {
            max-width: 700px;
            margin: 20px auto;
            padding: 20px;
            background-color: rgba(0,0,0,0.6);
            border-radius: 10px;
        }

        label {
            display: block;
            margin: 10px 0 5px;
            font-weight: bold;
        }

        input, select {
            width: 100%;
            padding: 8px;
        }

        button {
            margin-top: 15px;
            padding: 10px 20px;
            background-color: #0078d7;
            color: white;
            border: none;
            border-radius: 5px;
        }

        .results {
            margin-top: 20px;
            padding: 15px;
            background: rgba(255,255,255,0.3);
            border-radius: 8px;
        }
    </style>
</head>
<body class="bg-image">
    <div class="container">
        <h1>Abalone Age Predictor</h1>

        {% if result %}
        <div class="results">
            <h2>Prediction Results:</h2>
            <p><strong>Predicted Rings:</strong> {{ result.rings }}</p>
            <p><strong>Estimated Age:</strong> {{ result.age }} years</p>
            <p><strong>Water Type:</strong> {{ result.water }}</p>
            <p><strong>Number of Pearls:</strong> {{ result.pearls }}</p>
        </div>
        {% endif %}

        <form method="POST">
            <label>Sex:</label>
            <select name="sex" required>
                <option value="M">M</option>
                <option value="F">F</option>
                <option value="I">I</option>
            </select>

            <label>Length:</label>
            <input type="number" step="0.01" name="length" required>

            <label>Diameter:</label>
            <input type="number" step="0.01" name="diameter" required>

            <label>Height:</label>
            <input type="number" step="0.01" name="height" required>

            <label>Whole Weight:</label>
            <input type="number" step="0.01" name="whole_weight" required>

            <button type="submit">Predict</button>
        </form>
    </div>
</body>
</html>
"""

# -----------------------------------
# ROUTE
# -----------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        try:
            sex = request.form["sex"].upper()
            length = float(request.form["length"])
            diameter = float(request.form["diameter"])
            height = float(request.form["height"])
            whole_weight = float(request.form["whole_weight"])

            sex_value = le_sex.transform([sex])[0] if sex in le_sex.classes_ else 0

            input_df = pd.DataFrame(
                [[sex_value, length, diameter, height, whole_weight]],
                columns=X.columns
            )

            input_scaled = scaler.transform(input_df)

            predicted_rings = int(round(model_rings.predict(input_scaled)[0]))
            predicted_age = predicted_rings

            predicted_water = (
                "Marine" if predicted_rings > 10 or whole_weight > 1.0
                else "Freshwater"
            )

            if predicted_water == "Freshwater":
                pearls = max(1, int((predicted_rings + whole_weight + diameter)/1.5))
            else:
                pearls = max(1, int((predicted_rings + whole_weight + diameter)/3))

            result = {
                "rings": predicted_rings,
                "age": predicted_age,
                "water": predicted_water,
                "pearls": pearls
            }

        except Exception:
            traceback.print_exc()
            result = {
                "rings": "Error",
                "age": "Error",
                "water": "Error",
                "pearls": "Error"
            }

    return render_template_string(html_template, result=result)

# -----------------------------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
