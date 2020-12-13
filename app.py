import dash
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import dash_html_components as html

import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
from skimage import io

import base64
import numpy as np
import cv2

from generate_answer_key import generate_answer_key

from generate_perturbations import generate_mesh

app = dash.Dash(__name__, external_stylesheets=['assets/styles.css'])
app.title='Paper Flattening'


@app.callback([Output('app-content', 'children')],
              Input('upload-data', 'contents'),
              Input('section-tabs','value'),
              Input('perturbations','value'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'),
            )
def update_output(image_contents,section, perturbations, names, dates):
    CONTENT = html.Div()
    
    if image_contents is None:
        img = cv2.imread('./MathWorksheets/0007.png')
        
    else:
        
        im_bytes = base64.b64decode(image_contents[22:])
        im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    
        img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)

    if section == 'grade':
        # Run Detection Algorithm
        img_rgb, clean_output_img_rgb = generate_answer_key(img)
        
        

        # Convert to Base64: Answer Key
        jpg_img = cv2.imencode('.jpg', clean_output_img_rgb)#img_rgb)
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

        if perturbations == 2:
            try: 
                x, y, z = np.load('2_perturbations.npy')
                print('Here!')
            except:
                x, y, z = generate_mesh(n_mesh=n_mesh)
        else:
            x, y, z = generate_mesh(n_mesh=n_mesh)

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
                        scene_camera=dict(eye=dict(x=0., y=0., z=4.0)),
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
                                            html.P('About dataset generation'),
                                            html.H4('Choose Number of Perturbations'),
                                            dcc.Slider(
                                                min=0,
                                                max=9,
                                                marks={i: '{}'.format(i) for i in range(10)},
                                                value=2,
                                                id='perturbations',persistence_type='session',
                                            )  

                                        ],label='Dataset',value='dataset',className='section-tab',
                                    ),
                                    dcc.Tab(
                                        [
                                            html.H4('Mathworksheet Grading'),
                                            html.P('About math worksheet grading using template matching'),

                                        ],label='Grading',value='grade',className='section-tab',
                                    ),
                                ], id='section-tabs',value='dataset',persistence_type='session',
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
    app.run_server(debug=True)