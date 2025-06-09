from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="YOLO FastAPI Service")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "Bienvenido a la API de YOLO (demo)"}

@app.get("/predict")
def predict():
    # Demo endpoint
    return JSONResponse(content={"result": "Aquí iría la inferencia de YOLO (demo)"})
