from fastai.vision import load_learner, open_image

from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

import numpy as np
import uvicorn
import sys
import asyncio
import async
from pathlib import Path
import aiohttp
from io import BytesIO

from labels import labels

def load_model():
    
    try:
        learn = load_learner('./', 'trained_model.pkl')
        return learn
    except:
        print('An error occurred while loading the model')



app = Starlette()
app.debug = True
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='static'))





@app.route('/')
async def homepage(request):
    path = Path('./view/index.html')
    return HTMLResponse(path.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    
    learn = load_model()
    img_data = await request.form()
    img_bytes = await img_data['file'].read()
    img = open_image(BytesIO(img_bytes))
    
    prediction, probabilities = learn.predict(img)[0],learn.predict(img)[2]
    index = np.int(np.str(prediction))
    probabilities = np.float(max(probabilities) * 100)
    return JSONResponse({'result':str(labels[index])})
    
    
    
if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='82.48.66.71', port=5000)
