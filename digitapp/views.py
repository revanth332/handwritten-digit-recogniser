from django.shortcuts import render
from django.http import JsonResponse
from PIL import Image,ImageChops
import numpy as np
import tempfile
import joblib
import base64
import matplotlib.pyplot as plt

def pixalize_img(img_path):
    image = Image.open(img_path,"r")
    image = image.resize((28,28))
    px_data = list(image.getdata())
    px_data_latest = []
    for item in px_data:
        px_data_latest.append(sum(item)/255)

    # image = Image.open(img_path,'r')
    # image = image.convert('L')
    # image = ImageChops.invert(image)
    # image = image.resize((28,28))
    # px_data = list(image.getdata())
    # for i in range(len(px_data)):
    #     if px_data[i]/255 <= 0.43:
    #         px_data[i] = 0
    # px_data = np.array(px_data)/255
    # image.show(px_data)
    # image.show(px_data_latest)
    return px_data_latest

def inputPage(request):
    if request.method == "POST":
        cls = joblib.load('digit_model.sav')
        image_data = request.POST.get("image")
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(base64.b64decode(image_data.split(',')[1]))
                temp_file_path = temp_file.name
        image_data = pixalize_img(temp_file_path)
        ans = cls.predict([image_data])
        return JsonResponse({"success":True,"ans":str(ans[0])})
    else:
        return render(request,"index.html")
