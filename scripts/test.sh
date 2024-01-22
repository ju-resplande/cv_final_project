#!/bin/bash

DATA_DIR=data
backbones=("mit_b0" "efficientnetv2_b0")
epsilons_train=(0 0.1 0.2 0.25 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
epsilons_test=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
datasets=("test_split" 0 1 2 3)

for DATASET in ${datasets[@]}; do
    for BACKBONE in ${backbones[@]}; do
        for EPSILON_TRAIN in ${epsilons_train[@]}; do
            MODEL_DIR=output/${BACKBONE}_${EPSILON_TRAIN}
            REPORT_DIR=${MODEL_DIR}/eval
            mkdir -p ${REPORT_DIR}
            for EPSILON_TEST in ${epsilons_test[@]}; do
                python main.py evaluate ${MODEL_DIR}/model.pkl ${DATA_DIR}/${DATASET} ${REPORT_DIR}/${DATASET}_${EPSILON_TEST}.json --epsilon=${EPSILON_TEST}
            done
            python main.py evaluate ${MODEL_DIR}/model.pkl ${DATA_DIR}/${DATASET} ${REPORT_DIR}/${DATASET}_0.json
        done
    done
done