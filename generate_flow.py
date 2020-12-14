import numpy as np
import argparse


parser = argparse.ArgumentParser()
# For parsing commandline arguments
parser.add_argument("--meshArray", type=str, default='./perturb_mesh.npy')

args = parser.parse_args()

def get_flow(pmesh):
    mr = 850
    mc = 850
    tr = 850
    tc = 850
    mg = 4

    mh = 850
    mw = 850

    mx = np.linspace(mh, 1, mr,endpoint=False)
    my = np.linspace(1, mw, mc,endpoint=False)

    [my, mx] = np.meshgrid(mx, my)
    smesh = np.array([mx.flatten(), my.flatten()])

    px = pmesh[:, 0]
    py = pmesh[:, 1]

    minx = np.min(px.flatten())
    maxx = np.max(px.flatten())
    miny = np.min(py.flatten())
    maxy = np.max(py.flatten())

    px = (px - minx) / (maxx - minx)
    py = (py - miny) / (maxy - miny)

    px = px * (tc - 2 * mg - 1) + 1 + mg
    py = py * (tr - 2 * mg - 1) + 1 + mg
    pmesh = np.array([px, py])
    # Run LocalWarp C++ Compiled App
    fm = localwarp(pmesh, smesh, tr, tc, mr, mc)

    fm = np.reshape(fm, 2, tr, tc)
    fm = permute(fm, [2, 3, 1])

    # c++ and python 0/1 offset issue
    fm = np.array([fm[2:, :, :], fm[1, :, :]])
    fm = np.array([ fm[:, 2 :, :], fm[:, 1, :]])

    fm = fm.astype(float)

    return fm
if __name__ == '__main__':
    # Get Perturb Mesh Array from File
    pmesh = np.load(args.meshArray)

    image_flow = get_flow(pmesh)

