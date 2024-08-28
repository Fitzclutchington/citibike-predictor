from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from app.routers import predict

app = FastAPI()

templates = Jinja2Templates(directory="app/templates/")

app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("homepage.html", {"request": request})


app.include_router(predict.router)
