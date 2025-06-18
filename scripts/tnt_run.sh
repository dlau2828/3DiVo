# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

PATH_TO_OFFICIAL_TNT="data/TnT"
PATH_TO_PREPROC_TNT="data/TnT/TNT_GOF"

lanuch_exp() {
    local scene_name="$1"
    shift
    local output_dir="$1"
    shift
    local exp_args="$*"

    python train.py --cfg_files cfg/tnt_mesh.yaml --source_path $PATH_TO_PREPROC_TNT/TrainingSet/$scene_name/ --model_path $output_dir/$scene_name $exp_args
    python render.py $output_dir/$scene_name --skip_test --eval_fps
    python render.py $output_dir/$scene_name --skip_test --rgb_only --use_jpg
    python render_fly_through.py $output_dir/$scene_name/
    python extract_mesh.py $output_dir/$scene_name/ --save_gpu --use_vert_color --mesh_fname mesh_svr \
        --bbox_path $PATH_TO_OFFICIAL_TNT/$scene_name/"$scene_name"_mesh_bbox.txt \
        --final_lv 11
    python scripts/eval_tnt/run.py --dataset-dir $PATH_TO_OFFICIAL_TNT/$scene_name/ --traj-path $PATH_TO_PREPROC_TNT/TrainingSet/$scene_name/"$scene_name"_COLMAP_SfM.log --ply-path $output_dir/$scene_name/mesh/latest/mesh_svr.ply
    rm -r $output_dir/$scene_name/checkpoints/
}

ulimit -n 2048  # Increase maximum number of files the script can read

for scene in Barn Caterpillar Ignatius Truck Meetingroom Courthouse
do
    echo "============ start " $scene " ============"
    lanuch_exp $scene $1 "${@:2}"
    echo "============ end " $scene " ============"
done
