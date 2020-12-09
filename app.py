import dash
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import dash_html_components as html
import plotly.express as px
import pandas as pd
from skimage import io

import base64
import numpy as np
import cv2

from generate_answer_key import generate_answer_key

app = dash.Dash(__name__, external_stylesheets=['assets/styles.css'])


@app.callback([Output('input-worksheet', 'src'),
              Output('graded-worksheet', 'src')],
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(contents, names, dates):
    
    if contents is not None:
        
        im_bytes = base64.b64decode(contents[22:])
        im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
        
        img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
        # Run Detection Algorithm
        img_rgb, clean_output_img_rgb = generate_answer_key(img)
        
        

        # Convert to Base64: Answer Key
        jpg_img = cv2.imencode('.jpg', img_rgb)
        encoded_image = base64.b64encode(jpg_img[1]).decode('utf-8')
        img_base64 = str.format('data:image/png;base64,{}',encoded_image)

        # # Convert to Base64: Detections
        # jpg_img = cv2.imencode('.jpg', clean_output_img_rgb)
        # encoded_image = base64.b64encode(jpg_img[1]).decode('utf-8')
        # detections_img = str.format('data:image/png;base64,{}',encoded_image)
        
    else: 
        contents = ''
        detections_img = ''
        img_base64 = ''

        
        
    return [contents,img_base64]




app.layout = html.Div(children=
[
    html.H1(
        children='EECS 504 Project: Answer Key Generation App!',
        style={
            'textAlign': 'center'
        }
    ),

    html.Div(
            [
            html.Div(
                [
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        'Drag and Drop ',
                        html.A('a Math Worksheet')
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },
                    # Allow multiple files to be uploaded
                    multiple=False
                ),
                
            ],className='columns six',
            )
                
        ],className="row flex-display",
    ),

    html.Div(
            [
                html.Div(
                    [

                    html.Img(id="input-worksheet",style={'width':'100%'}),
                    #print(contents)
                

                    ],className="pretty_container three columns",
                    
                    
                ),
                # html.Div(
                #     [
                #         html.Img(id="detections",style={'width':'100%'}),

                #     ],
                    
                #     className=" pretty_container three columns",
                # ),
                html.Div(
                    [
                        html.Img(id="graded-worksheet",style={'width':'100%'}),

                    ],
                    
                    className=" pretty_container three columns",
                ),
                
            ],
            className="row flex-display",
    ),
    html.Div(id='output-image-upload'),
    

],
id="mainContainer",
style={"display": "flex", "flex-direction": "column"},
)

if __name__ == '__main__':
    app.run_server(debug=True)