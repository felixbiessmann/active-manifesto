# manifesto model

this folder should contain all logic for the actual training pipeline
of the manifesto model and the http facade for communicating
with the user interface component.

## Initial Model
Only trains simple linear model and predicts left/right/neutral when a POST request comes in

```
cd services/manifesto_model
docker build -t  manifesto_model .
docker run -p 0.0.0.0:5000
curl -d "text=sicherheit" http://0.0.0.0:5000/predict
```
