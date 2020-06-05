#!/bin/bash

## Interpolation
#echo "Rendering interpolation ..."
#echo "[Airplanes] "
#python utils/visualize_points.py \
#    drive/airplane/pointnet/interpolation/interpolation_samples.pth \
#    renders/airplane/pointnet/interpolation/ \
#    --torch \
#    --batch
#echo "[Car] "
#python utils/visualize_points.py \
#    drive/car/pointnet/interpolation/interpolation_samples.pth \
#    renders/car/pointnet/interpolation/ \
#    --torch \
#    --batch
#echo "[Chair] "
#python utils/visualize_points.py \
#    drive/chair/pointnet/interpolation/interpolation_samples.pth \
#    renders/chair/pointnet/interpolation/ \
#    --torch \
#    --batch
#
#echo "Rendering samples ..."
## Samples visualization
#echo "[Airplane]"
#python utils/visualize_points.py \
#    drive/airplane/pointnet/samples/may_30/metrics_samples10_0.pth \
#    renders/airplane/pointnet/samples/ \
#    --torch \
#    --batch
#
#echo "[Car]"
#python utils/visualize_points.py \
#    drive/car/pointnet/samples/may_30/metrics_samples125_0.pth \
#    renders/car/pointnet/samples/ \
#    --torch \
#    --batch
#
#echo "[Chair]"
#python utils/visualize_points.py \
#    drive/chair/pointnet/samples/metrics_samples11_0.pth \
#    renders/chair/pointnet/samples/ \
#    --torch \
#    --batch
#
## Test reconstruction
#echo "Rendering test reconstructions ..."
#echo "[Airplane]"
#python utils/visualize_points.py \
#    drive/airplane/pointnet/reconstructions/reconstructions_val.pth \
#    renders/airplane/pointnet/val_reconstructions/ \
#    --torch \
#    --batch
#echo "[Car]"
#python utils/visualize_points.py \
#    drive/car/pointnet/reconstructions/reconstructions_val.pth \
#    renders/car/pointnet/val_reconstructions/ \
#    --torch \
#    --batch
#echo "[Chair]"
#python utils/visualize_points.py \
#    drive/chair/pointnet/reconstructions/reconstructions_val.pth \
#    renders/chair/pointnet/val_reconstructions/ \
#    --torch \
#    --batch

#echo "Rendering train reconstructions ..."
# Train reconstruction
#echo "[Airplane]"
#python utils/visualize_points.py \
#    drive/airplane/pointnet/reconstructions/reconstructions_train.pth \
#    renders/airplane/pointnet/train_reconstructions/ \
#    --torch \
#    --batch
#echo "[Car]"
#python utils/visualize_points.py \
#    drive/car/pointnet/reconstructions/reconstructions_train.pth \
#    renders/car/pointnet/train_reconstructions/ \
#    --torch \
#    --batch
#echo "[Chair]"
#python utils/visualize_points.py \
#    drive/chair/pointnet/reconstructions/reconstructions_train.pth \
#    renders/chair/pointnet/train_reconstructions/ \
#    --torch \
#    --batch
#
#echo "Rendering train references ..."
#echo "[Airplane]"
#python utils/visualize_points.py \
#    drive/airplane/pointnet/reconstructions/references_train.pth \
#    renders/airplane/pointnet/train_references/ \
#    --torch \
#    --batch
#echo "[Car]"
#python utils/visualize_points.py \
#    drive/car/pointnet/reconstructions/references_train.pth \
#    renders/car/pointnet/train_references/ \
#    --torch \
#    --batch
#echo "[Chair]"
#python utils/visualize_points.py \
#    drive/chair/pointnet/reconstructions/references_train.pth \
#    renders/chair/pointnet/train_references/ \
#    --torch \
#    --batch

#echo "Rendering test references ..."
#echo "[Airplane]"
#python utils/visualize_points.py \
#    drive/airplane/pointnet/reconstructions/references_val.pth \
#    renders/airplane/pointnet/val_references/ \
#    --torch \
#    --batch
#echo "[Car]"
#python utils/visualize_points.py \
#    drive/car/pointnet/reconstructions/references_val.pth \
#    renders/car/pointnet/val_references/ \
#    --torch \
#    --batch
#echo "[Chair]"
#python utils/visualize_points.py \
#    drive/chair/pointnet/reconstructions/references_val.pth \
#    renders/chair/pointnet/val_references/ \
#    --torch \
#    --batch

echo "Rendering common-rare ..."
echo "[Airplane]"
python utils/visualize_points.py \
    drive/airplane/pointnet/common-rare/common_clouds.pth \
    renders/airplane/pointnet/common/ \
    --torch \
    --batch
python utils/visualize_points.py \
    drive/airplane/pointnet/common-rare/rare_clouds.pth \
    renders/airplane/pointnet/rare/ \
    --torch \
    --batch
echo "[Car]"
python utils/visualize_points.py \
    drive/car/pointnet/common-rare/common_clouds.pth \
    renders/car/pointnet/common/ \
    --torch \
    --batch
python utils/visualize_points.py \
    drive/car/pointnet/common-rare/rare_clouds.pth \
    renders/car/pointnet/rare/ \
    --torch \
    --batch
echo "[Chair]"
python utils/visualize_points.py \
    drive/chair/pointnet/common-rare/common_clouds.pth \
    renders/chair/pointnet/common/ \
    --torch \
    --batch
python utils/visualize_points.py \
    drive/chair/pointnet/common-rare/rare_clouds.pth \
    renders/chair/pointnet/rare/ \
    --torch \
    --batch
