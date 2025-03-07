from django.shortcuts import render
from django.conf import settings
import os
import pickle
import sklearn
import pandas as pd

# Create your views here.


def home(request):
    predict = None
    model_path = os.path.join(settings.BASE_DIR, 'pkl\lrmodel.pkl')
    encoder_path = os.path.join(settings.BASE_DIR, 'pkl\ordinalencoder.pkl')
    scaler_path = os.path.join(settings.BASE_DIR, 'pkl\standardscaler.pkl')

    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    with open(encoder_path, 'rb') as file:
        ordinal = pickle.load(file)

    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)


    if request.method == "POST":
        brand = request.POST['brand']
        car_model = request.POST['model']
        model_year = int(request.POST['model_year'])
        mileage = int(request.POST['mileage'])
        fuel_type = request.POST['fuel_type']
        transmission = request.POST['transmission']
        hp = float(request.POST['hp'])
        engine_size = float(request.POST['engine_size'])
        accident = request.POST['accident']
        clean_title = request.POST['clean_title']

        data = {
            "brand":brand,
            "model":car_model,
            "model_year":model_year,
            "milage":mileage,
            "fuel_type":fuel_type,
            "transmission":transmission,
            "accident":accident,
            "clean_title":clean_title,
            "hp":hp,
            "l":engine_size,
        }

        dataset = pd.DataFrame([data])
        categorical_data = ['brand', 'model', 'fuel_type', 'transmission', 'accident', 'clean_title']
        dataset[categorical_data] = ordinal.transform(dataset[categorical_data])
        dataset = scaler.transform(dataset)
        predict = model.predict(dataset)

    return render(request, "index.html", {'predict': predict})