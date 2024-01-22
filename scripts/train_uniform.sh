#!/bin/bash

backbones=("mit_b0" "efficientnetv2_b0")

for BACKBONE in ${backbones[@]}; do
    #OUTPUT_DIR=output/${BACKBONE}_uniform
    #python main.py train ${BACKBONE} ${OUTPUT_DIR} --uniform="pixel"
    OUTPUT_DIR=output/${BACKBONE}_uniform_image
    python main.py train ${BACKBONE} ${OUTPUT_DIR} --uniform=image
done