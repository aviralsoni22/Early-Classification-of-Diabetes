# import library
import uvicorn
from fastapi import FastAPI
from Diabetes import Dia
import pickle
#create an api object
app=FastAPI()
pickle_in = open("D://VIT Vellore//ML//Early Classification of Diabetes//Diabetes.pkl","rb")
classifier = pickle.load(pickle_in)
#Index route, opens atomatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return{'message': 'Welcome!'}

#Route with a single parameter, returns the parameter within a message 
#located at : http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name:str):
    return{'message': f'Hello, {name}'}

# Route with http://127.0.0.1:8000/docs
@app.post('/predict')
def predict_diabetes(data:Dia):
    data=data.dict()
    age =  data['age']
    gender = data['gender']
    polydipsia = data['polydipsia']
    sudden_weight_loss = data['sudden_weight_loss']
    weakness = data['weakness']
    polyphagia = data['polyphagia']
    genital_thrush = data['genital_thrush']
    visual_blurring = data['visual_blurring']
    itching = data['itching']
    irritability = data['irritability']
    delayed_healing = data['delayed_healing']
    partial_paresis = data['partial_paresis']
    muscle_stiffness = data['muscle_stiffness']
    alopecia = data['alopecia']
    obesity = data['obesity']
    prediction = classifier.predict([[age,gender,polydipsia,sudden_weight_loss,weakness,polyphagia,genital_thrush,visual_blurring,itching,irritability,delayed_healing,partial_paresis,muscle_stiffness,alopecia,obesity]])
    if(prediction == 1):
        prediction = "Diabetic"
    else:
        prediction = "Not Diabetic"
    return{
        'prediction' : prediction
    }
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port = 8000)
#uvicorn app:app --reload