import dash
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_daq as daq


import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
from skimage import io

import base64
import numpy as np
import cv2

from generate_answer_key import generate_answer_key_main

from generate_perturbations import generate_mesh

app = dash.Dash(__name__, external_stylesheets=['assets/styles.css'])
app.title='Paper Flattening'
server = app.server

@app.callback([Output('app-content', 'children')],
              Input('upload-data', 'contents'),
              Input('section-tabs','value'),
              Input('perturbations','value'),
              Input('show-detections','on'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'),
            )
def update_output(image_contents,section, perturbations, show_detections, names, dates):
    CONTENT = html.Div()
    
    if image_contents is None:
        # Load in a Default Worksheet if there is no curretn upload
        img = cv2.imread('./MathWorksheets/0007.png')
        
    
        jpg_img = cv2.imencode('.jpg', img)
        encoded_image = base64.b64encode(jpg_img[1]).decode('utf-8')
        image_contents = str.format('data:image/png;base64,{}',encoded_image)

        
    im_bytes = base64.b64decode(image_contents[22:])
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array

    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)

    if section == 'grade':
        # Run Detection Algorithm
        img_rgb, clean_output_img_rgb = generate_answer_key_main(img)
        print(show_detections)
        if show_detections:
            
            img2show = img_rgb
        else:
            img2show = clean_output_img_rgb

        # Convert to Base64: Answer Key
        jpg_img = cv2.imencode('.jpg', img2show)#img_rgb)
        encoded_image = base64.b64encode(jpg_img[1]).decode('utf-8')
        img_base64 = str.format('data:image/png;base64,{}',encoded_image)

        # # Convert to Base64: Detections
        # jpg_img = cv2.imencode('.jpg', clean_output_img_rgb)
        # encoded_image = base64.b64encode(jpg_img[1]).decode('utf-8')
        # detections_img = str.format('data:image/png;base64,{}',encoded_image)

        CONTENT = html.Div(
                    [
                        html.Div(
                            [

                                html.Img(id="input-worksheet",style={'width':'100%'},src=image_contents),
                                #print(contents)
                        

                            ],className="three columns",
                            
                            
                        ),
                        html.Div(
                            [
                                html.Img(id="graded-worksheet",style={'width':'100%'},src=img_base64),

                            ],
                            
                            className="three columns",
                        ),
                    ],className="row flex-display",
                )
    elif section == 'dataset':
        n_mesh = 500

        
        x, y, z = generate_mesh(n_mesh=n_mesh,perturbations=perturbations)

        # resize image
        resized = cv2.resize(img, (n_mesh,n_mesh), interpolation = cv2.INTER_AREA)

        print('Resized Dimensions : ',resized.shape)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        ambient = 0.6
        diffuse = 0.8
        specular =0.9
        specular_exp = 25
        lighting_effects = dict(ambient=ambient, diffuse=diffuse, roughness = 0., specular=specular, fresnel=5)
        
        fig = go.Figure(data=[go.Surface(z=z, x=x, y=y,
                                        showscale=False,
                                        surfacecolor=np.flipud(np.rot90(gray,-1)),
                                        colorscale='gray',
                                        lighting=lighting_effects)])
        fig.update_layout( autosize=True,
                        scene_camera=dict(eye=dict(x=0., y=0., z=3.0)),
                        scene_aspectmode='data',
                        paper_bgcolor='rgba(0,0,0,0)',
                        scene=dict(
                            xaxis_visible=False,
                            yaxis_visible=False, 
                            zaxis_visible=False,
                            ),
                        
                        width=900, height=900,
                        #margin=dict(l=65, r=50, b=65, t=90)
                        )


        CONTENT = html.Div([
                                dcc.Graph(figure=fig)
                            ])
                                
                                
                            


        
        
    

        
        
    return [CONTENT]


def layout():
    return html.Div(
    [
        
        html.Div(
                [

                    html.H1(
                        
                        children='EECS 504 Project: The Flatteners',
                        id ='title',
                        style={
                            'textAlign': 'center'
                        }
                     ),
                     dcc.Loading(
                         [
                             html.Div(id='app-content'),

                         ]
                     )
                    
                    
                    
                    
                    
                ],
                
                id='app-container',
        ),
        #html.Div(id='output-image-upload'),
        html.Div(
                [
                
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select a Math Worksheet')
                        ]),
                        # No CSS alternative here
                        style={
                            "color": "darkgray",
                            "width": "100%",
                            "height": "50px",
                            "lineHeight": "50px",
                            "borderWidth": "1px",
                            "borderStyle": "dashed",
                            "borderRadius": "5px",
                            "borderColor": "darkgray",
                            "textAlign": "center",
                            "padding": "2rem 0",
                            "margin-bottom": "2rem",
                        },
                        
                        # Allow multiple files to be uploaded
                        multiple=False
                    ),
                    html.Div(
                        [
                            dcc.Tabs(
                                [

                                    
                                    dcc.Tab(
                                        [
                                            html.H4('Dataset Generation'),
                                            html.P(['This application demonstrates the generation of realistic creases and folds given a flattened image of a document. It takes in an image of a document and outputs that same image with creases and folds. These types of images were used to learn the mapping of pixels between a flattened document and its creased equivalent. Please check out our ', 
                                            html.A('Github',href ='https://github.com/arnavsharma/the-flatteners'),' for more information!']),
                                            html.H4('Choose Number of Creases'),
                                            dcc.Slider(
                                                min=0,
                                                max=9,
                                                marks={i: '{}'.format(i) for i in range(10)},
                                                value=2,
                                                id='perturbations',persistence_type='session',persistence=True,
                                            )  

                                        ],label='Dataset',value='dataset',className='section-tab',
                                    ),
                                    dcc.Tab(
                                        [
                                            html.H4('Math Worksheet Grading'),
                                            html.P(['The math worksheet answer key has been generated using template matching and K-means clustering. If you would like to try out a different worksheet please check out our ', html.A('GitHub.',href='https://github.com/arnavsharma/the-flatteners/tree/main/MathWorksheets'),
                                            ' If you see an error page, please refresh the page.']),
                                            html.H4('Turn on Detections'),
                                            daq.BooleanSwitch(id='show-detections',persistence_type='session',persistence=True,color= '#3377FF',
                                                on=False),


                                        ],label='Grading',value='grade',className='section-tab',
                                    ),
                                ], id='section-tabs',value='dataset',persistence_type='session',persistence=True,
                            )

                        ], className='control-tabs',
                    )
                    
                
                    
                ],id ='sidebar',
        ),
        

    ],
    id="root",
    
    )

# Add Layout to App
app.layout = layout()

if __name__ == '__main__':
    # Run Applciation
    app.run_server(debug=False,host='0.0.0.0')