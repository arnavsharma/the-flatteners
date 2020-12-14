DATASETLOC=./ForAustin5

echo $DATASETLOC
python local_patch.py --dataPath $DATASETLOC --savePath ./dataset_patch --res 256 --totalNum 2700 --testNum 540 
echo 'Finished Creating Local Patches'
python global_patch.py --dataPath $DATASETLOC --savePath ./dataset_patch --res 256 --totalNum 2700 --testNum 540
echo 'Finished Creating Global Patches'
python train.py --dataset_dir ./dataset_patch --savemodel_dir 'model_4Austin5_0_0001_LR_noTransfer.pkl' --batch_size 4
echo 'Finished Training'