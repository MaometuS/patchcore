
datapath=/home/maometus/Documents/datasets/real_iad
datasets=("audiojack" "bottle_cap" "button_battery" "end_cap" "eraser" "fire_hood" "mint" "mounts" "pcb" "phone_battery" "plastic_nut" "plastic_plug" "porcelain_doll" "regulator" "rolled_strip_base" "sim_card_set" "switch" "tape" "terminalblock" "toothbrush" "toy" "toy_brick" "transistor1" "u_block" "usb" "usb_adaptor" "vcpill" "wooden_beads" "woodstick" "zipper")
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

env PYTHONPATH=src python bin/run_patchcore.py --gpu 0 --seed 22 --save_patchcore_model --log_group IM320_CUSTOM_GRID_WD50_L2-3_P001_D1024-1024_PS-5_AN-3_S0 --log_project Real_IAD_Results results \
patch_core -b custom_wd50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 3 --patchsize 5 sampler -p 0.01 approx_greedy_coreset dataset --resize 366 --imagesize 320 "${dataset_flags[@]}" real_iad $datapath