from database import Base, engine
from fastapi import FastAPI
from parser import create_protocol_router_llm
#from upload import upload_router

app = FastAPI()
Base.metadata.create_all(bind=engine)

#app.include_router(upload_router)
app.include_router(create_protocol_router_llm("cough"))
app.include_router(create_protocol_router_llm("back pain"))
app.include_router(create_protocol_router_llm("fever"))
app.include_router(create_protocol_router_llm("headache"))
app.include_router(create_protocol_router_llm("neck pain"))
app.include_router(create_protocol_router_llm("ear pain"))
app.include_router(create_protocol_router_llm("fatigue"))
app.include_router(create_protocol_router_llm("diarrhea"))
app.include_router(create_protocol_router_llm("swollen glands"))
app.include_router(create_protocol_router_llm("chest pain"))
app.include_router(create_protocol_router_llm("shortness of breath"))
app.include_router(create_protocol_router_llm("anxiety or stress"))
app.include_router(create_protocol_router_llm("insomnia or sleep disturbance"))
app.include_router(create_protocol_router_llm("urinary discomfort"))
app.include_router(create_protocol_router_llm("knee joint pain"))
app.include_router(create_protocol_router_llm("skin rashes"))
app.include_router(create_protocol_router_llm("sore throat"))
app.include_router(create_protocol_router_llm("dizziness"))
app.include_router(create_protocol_router_llm("nausea and vomiting"))
app.include_router(create_protocol_router_llm("abdominal pain"))
app.include_router(create_protocol_router_llm("allergic reactions"))



@app.get("/")
def root():
    return {"message": "Welcome to the Digi Vaidya Backend Server"}