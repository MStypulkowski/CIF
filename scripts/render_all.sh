#!/bin/bash

python -m utils.animate_points \
    data/to_animate/samples_airplane.pth \
    data/animations/airplane/ \
    --indices 8 41 59 64

python -m utils.animate_points \
    data/to_animate/samples_car.pth \
    data/animations/car/ \
    --indices 8 49 57 96 98

python -m utils.animate_points \
    data/to_animate/samples_chair.pth \
    data/animations/chair/ \
    --indices 26 43 67 83 91

python utils/images_to_vid.py data/animations/airplane/8 
python utils/images_to_vid.py data/animations/airplane/41
python utils/images_to_vid.py data/animations/airplane/59
python utils/images_to_vid.py data/animations/airplane/64

python utils/images_to_vid.py data/animations/car/8 
python utils/images_to_vid.py data/animations/car/49
python utils/images_to_vid.py data/animations/car/57
python utils/images_to_vid.py data/animations/car/96
python utils/images_to_vid.py data/animations/car/98

python utils/images_to_vid.py data/animations/car/26 
python utils/images_to_vid.py data/animations/car/43
python utils/images_to_vid.py data/animations/car/67
python utils/images_to_vid.py data/animations/car/83
python utils/images_to_vid.py data/animations/car/91