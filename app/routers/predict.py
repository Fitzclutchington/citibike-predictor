from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from data_utils.predict import predict_ebikes


templates = Jinja2Templates(directory="app/templates/")

router = APIRouter(
    prefix="/predict", tags=["predict"], responses={404: {"description": "Not found"}}
)


@router.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse(
        "predict_form.html", {"request": request, "prediction": None}
    )


@router.post("/", response_class=HTMLResponse)
async def predict(
    request: Request,
    date: str = Form(...),
    lat: float = Form(...),
    lon: float = Form(...),
    capacity: int = Form(...),
):
    # Use the predict_ebikes function to get the prediction
    prediction = predict_ebikes(date, lat, lon, capacity)
    return templates.TemplateResponse(
        "predict_form.html", {"request": request, "prediction": f"{prediction:.2f}"}
    )
