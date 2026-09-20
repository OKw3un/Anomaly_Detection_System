from fastapi import FastAPI
from src.api.routers import datasets, analysis

app = FastAPI(
    title="Anomaly Detection Engine API",
    description="Farklı veri setleri üzerinde çoklu anomali algoritmalarını çalıştıran ve kesişimlerini bulan backend servisi.",
    version="1.0.0"
)

# Yazdığımız router'ları ana uygulamaya bağlıyoruz
app.include_router(datasets.router)
app.include_router(analysis.router)

@app.get("/")
def read_root():
    return {"message": "Anomaly Detection Engine çalışıyor. Dokümantasyon için /docs adresine gidin."}

if __name__ == "__main__":
    import uvicorn
    # Çalıştırmak için: python main.py
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)