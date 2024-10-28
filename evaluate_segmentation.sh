
datapath=/home/maometus/Documents/datasets/mvtec_anomaly_detection
loadpath=/home/maometus/Documents/projects/patchcore/results/MVTecAD_Results

modelfolder=IM320_CUSTOM_GRID_WD50_L2-3_P001_D1024-1024_PS-5_AN-3_S0
savefolder=evaluated_results'/'$modelfolder

# datasets=('bottle'  'cable'  'capsule'  'carpet'  'grid'  'hazelnut' 'leather'  'metal_nut'  'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
datasets=('grid')
model_flags=($(for dataset in "${datasets[@]}"; do echo '-p '$loadpath'/'$modelfolder'/models/mvtec_'$dataset; done))
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))

env PYTHONPATH=src python bin/load_and_evaluate_patchcore.py --gpu 0 --seed 0 $savefolder \
patch_core_loader "${model_flags[@]}" --faiss_on_gpu \
dataset --resize 366 --imagesize 320 "${dataset_flags[@]}" mvtec $datapath
