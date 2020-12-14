#!/bin/bash
#!/bin/expect
FILENAME=./ForAustin5/img/0005_5.png
#MODEL = 

echo $FILENAME
#'./docProjModels/model_geoNet.pkl'

# #C:/Users/AJEFFR16/AppData/Local/Continuum/Anaconda3/python.exe "c:/Users/AJEFFR16/Documents/School/EECS 504/Project/DocProj/eval.py" --
echo 'Running eval.py'
C:/Users/AJEFFR16/AppData/Local/Continuum/Anaconda3/python.exe eval.py --imgPath $FILENAME --modelPath './docProjModels/model_geoNet.pkl' --saveImgPath new_resized_image.png --saveFlowPath myflow.npy
echo 'Running Graph Cuts'
"C:/Users/AJEFFR16/Documents/School/EECS 504/Project/DocProj/Stitching/Graphcut.exe" 'C:\Users\AJEFFR16\Documents\School\EECS 504\Project\DocProj\myflow.npy' 'C:\Users\AJEFFR16\Documents\School\EECS 504\Project\DocProj\my_new_flow.npy'  
#expect "Press any key to continue . . ." { send "a" }
echo 'Running resampling.py'
C:/Users/AJEFFR16/AppData/Local/Continuum/Anaconda3/python.exe resampling.py --img_path new_resized_image.png --flow_path my_new_flow.npy 
echo 'Running eval_illumination.py'
C:/Users/AJEFFR16/AppData/Local/Continuum/Anaconda3/python.exe eval_illumination.py --imgPath result.png --savPath output.png --modelPath model_illNet.pkl
echo 'Done'