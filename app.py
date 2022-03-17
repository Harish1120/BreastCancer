from flask import Flask,render_template, request
import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=["post"])
def predict():
    radius_mean = request.form.get('radius_mean')
    texture_mean = request.form.get('texture_mean')
    perimeter_mean = request.form.get('perimeter_mean')
    area_mean = request.form.get('area_mean')
    smoothness_mean = request.form.get('smoothness_mean')
    compactness_mean = request.form.get('compactness_mean')
    concavity_mean = request.form.get('concavity_mean')
    concave_points_mean = request.form.get('concave_points_mean')
    symmetry_mean = request.form.get('symmetry_mean')
    fractal_dimension_mean = request.form.get('fractal_dimension_mean') 
    radius_se = request.form.get('radius_se')
    texture_se = request.form.get('texture_se')
    perimeter_se = request.form.get('perimeter_se')
    area_se = request.form.get('area_se')
    smoothness_se = request.form.get('smoothness_se')
    compactness_se = request.form.get('compactness_se')
    concavity_se = request.form.get('concavity_se')
    concave_points_se = request.form.get('concave_points_se')
    symmetry_se = request.form.get('symmetry_se')
    fractal_dimension_se = request.form.get('fractal_dimension_se')
    radius_worst = request.form.get('radius_worst')
    texture_worst = request.form.get('texture_worst')
    perimeter_worst = request.form.get('perimeter_worst')
    area_worst = request.form.get('area_worst')
    smoothness_worst = request.form.get('smoothness_worst')
    compactness_worst = request.form.get('compactness_worst')
    concavity_worst = request.form.get('concavity_worst')
    concave_points_worst = request.form.get('concave_points_worst')
    symmetry_worst = request.form.get('symmetry_worst')
    fractal_dimension_worst = request.form.get('fractal_dimension_worst')

    model = joblib.load('cancer_96.pkl')

    data = model.predict([[radius_mean,texture_mean, perimeter_mean,area_mean,
       smoothness_mean,compactness_mean,concavity_mean,
       concave_points_mean,symmetry_mean,fractal_dimension_mean,
       radius_se,texture_se,perimeter_se,area_se,smoothness_se,
       compactness_se,concavity_se,concave_points_se,symmetry_se,
       fractal_dimension_se,radius_worst,texture_worst,
       perimeter_worst,area_worst,smoothness_worst,
       compactness_worst,concavity_worst,concave_points_worst,
       symmetry_worst,fractal_dimension_worst]])

    print(data)
    if data[0] == 0:
        output = 'Patient Does Not Have Cancer'
    else:
        output = 'Patient is Diagnosed with Cancer'
    return render_template('prediction.html',output = output)

app.run(debug = True)








