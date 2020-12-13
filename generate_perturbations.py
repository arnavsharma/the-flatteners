import numpy as np
import numpy.matlib

from skimage import io
from scipy.io import savemat
import cv2

import plotly.graph_objects as go

import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plt

import argparse


parser = argparse.ArgumentParser()
# For parsing commandline arguments
parser.add_argument("--mathWorksheet", type=str, default='./MathWorksheets/0002.png', help='the math worksheet file')
parser.add_argument("--num_pertubations",type=int,default=2)
parser.add_argument("--mesh_size",type=int,default=500)
args = parser.parse_args()

def generate_mesh(perturbations=2,n_mesh=500):

    
    x = np.linspace(0, n_mesh,n_mesh,endpoint=False).astype(int)
    y = np.linspace(0, n_mesh,n_mesh,endpoint=False).astype(int)


    z_height = np.zeros((n_mesh,n_mesh)).astype(int)

    xv, yv = np.meshgrid(x, y)

    mesh_surface = np.array([xv.flatten(), yv.flatten(),z_height.flatten()]).T

    N = mesh_surface.shape[0]



    for p in range(perturbations):
        random_vertex_idx = np.random.randint(N)

        random_vertex = mesh_surface[random_vertex_idx,:]

        # Find the vector between all vertex and the selected one
        xv =  mesh_surface -  random_vertex

        mv  = (np.random.rand(3) - 0.5)*100
        

        mv[2] = (np.random.rand() - 0.5)*250
        print(mv)

        hmv = np.matlib.repmat(mv,xv.shape[0],1)

        d = np.cross(xv,hmv)


        d = abs(d[:,2]) /np.linalg.norm(mv)

        wt = d

        alpha = np.random.rand()* 100+ 100

        wt = alpha /(wt + alpha)

        msmv = np.outer(wt,mv)

        mesh_surface = mesh_surface + msmv


    x = mesh_surface[:,0].reshape((n_mesh,n_mesh))
    y = mesh_surface[:,1].reshape((n_mesh,n_mesh))
    z = mesh_surface[:,2].reshape((n_mesh,n_mesh))

    mesh = (x,y,z)

    return mesh

if __name__ == '__main__':
    # Grab Paramaters from Arguments
    n_mesh = args.mesh_size
    perturbations = args.num_pertubations

    path2File = args.mathWorksheet # GRab Path to File



    # Generate Mesh Grid of Pertubations
    x,y,z = generate_mesh(perturbations,n_mesh)

    # Read in Image
    img = cv2.imread(path2File, cv2.IMREAD_UNCHANGED)
    
    print('Original Dimensions : ',img.shape)

    # Resize Image to be teh same as the Mesh Grid
    resized = cv2.resize(img, (n_mesh,n_mesh), interpolation = cv2.INTER_AREA)
    
    print('Resized Dimensions : ',resized.shape)
    # Convert Image to GRay Scale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Set a Few Lighting Settings
    ambient = 0.6
    diffuse = 0.8
    specular =0.9
    specular_exp = 25
    lighting_effects = dict(ambient=ambient, diffuse=diffuse, roughness = 0., specular=specular, fresnel=5)
    # Plot the Image
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y,
                        showscale=False,
                        surfacecolor=np.flipud(np.rot90(gray,-1)),
                        colorscale='gray',
                        lighting=lighting_effects)])
    fig.update_layout( autosize=True,
                    scene_camera=dict(eye=dict(x=0., y=0., z=4.0)),
                    scene_aspectmode='data',
                    #paper_bgcolor='rgba(0,0,0,0)',
                    scene=dict(
                        xaxis_visible=False,
                        yaxis_visible=False, 
                        zaxis_visible=False,
                        ),
                    
                    width=900, height=900,
                    #margin=dict(l=65, r=50, b=65, t=90)
                    )
    fig.show()

