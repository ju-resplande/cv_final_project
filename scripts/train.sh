#!/bin/bash

backbones=("mit_b0" "efficientnetv2_b0")
epsilons=(0 0.1 0.2 0.25 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

for BACKBONE in ${backbones[@]}; do
    for EPSILON in ${epsilons[@]}; do
        OUTPUT_DIR=output/${BACKBONE}_${EPSILON}
        python main.py train ${BACKBONE} ${OUTPUT_DIR} --epsilon=${EPSILON}
    done
done