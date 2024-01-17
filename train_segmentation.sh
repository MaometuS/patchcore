
datapath=/home/maometus/Documents/datasets/mvtec_anomaly_detection
datasets=('bottle'  'cable'  'capsule'  'carpet'  'grid'  'hazelnut' 'leather'  'metal_nut'  'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

python bin/run_patchcore.py --gpu 0 --seed 22 --save_patchcore_model --log_group IM320_IAD_L2-3_P001_D1024-1024_PS-5_AN-3_S39 --log_project MVTecAD_Results results \
patch_core -b custom_iad -le blocks.10 -le blocks.11 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 3 --patchsize 5 sampler -p 0.01 approx_greedy_coreset dataset --resize 366 --imagesize 320 "${dataset_flags[@]}" mvtec $datapath