from flask import Flask, request, render_template
from src.DiamondPricePrediction.pipelines.prediction_pipeline import PredictPipeline, CustomData

print("imports done")  # ← add this

app = Flask(__name__)

print("flask app created")  # ← add this

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = CustomData(
        carat=float(request.form.get('carat')),
        cut=request.form.get('cut'),
        color=request.form.get('color'),
        clarity=request.form.get('clarity'),
        depth=float(request.form.get('depth')),
        table=float(request.form.get('table')),
        x=float(request.form.get('x')),
        y=float(request.form.get('y')),
        z=float(request.form.get('z'))
    )
    features = data.get_data_as_dataframe()
    pipeline = PredictPipeline()
    prediction = pipeline.predict(features)
    return render_template('result.html', price=round(prediction[0], 2))

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)