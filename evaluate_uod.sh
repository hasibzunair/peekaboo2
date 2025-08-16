MODEL=$1
DATASET_DIR=$2

# Single object discovery evaluation
for DATASET in VOC07 VOC12 COCO20k
do
    python evaluate.py --eval-type uod --dataset-eval $DATASET \
            --model-weights $MODEL --evaluation-mode single --dataset-dir $DATASET_DIR
done


