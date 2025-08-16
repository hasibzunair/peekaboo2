MODEL=$1
DATASET_DIR=$2
MODE=$3

# Unsupervised saliency detection evaluation
for DATASET in ECSSD DUTS-TEST DUT-OMRON
do
    python evaluate.py --eval-type saliency --dataset-eval $DATASET \
            --model-weights $MODEL --evaluation-mode $MODE --apply-bilateral --dataset-dir $DATASET_DIR
done


