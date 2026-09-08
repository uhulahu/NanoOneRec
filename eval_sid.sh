#!/bin/bash

DATASET="Industrial_and_Scientific"
ROOT="data/Amazon23/$DATASET"
SID_VARIANT="rqkmeans-td-last-20260903"
CODES_FILE="data/Amazon23/Industrial_and_Scientific/sid/rqkmeans-td-last-20260903/Industrial_and_Scientific.codes_constrained.npy"
INDEX_FILE="data/Amazon23/Industrial_and_Scientific/sid/rqkmeans-td-last-20260903/Industrial_and_Scientific.index.json"

python eval_sid.py --dataset $DATASET \
                --root $ROOT \
                --variant $SID_VARIANT \
                --codes_file $CODES_FILE \
                --index_file $INDEX_FILE \
                --k 256 \
                --l 3