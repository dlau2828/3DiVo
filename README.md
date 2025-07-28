# Sparse Voxels Rasterizer

To train run this:

python train.py --eval --source_path data/FOLDER --model_path result/FOLDER --lambda_T_inside 0.01 --lambda_normal_dmean 0.001 --lambda_normal_dmed 0.001 --lambda_ascending 0.01

All data has to be processed with colmap with pinhole beforehand. Darren has all of the data stored sepearetely because it was too large to push. 