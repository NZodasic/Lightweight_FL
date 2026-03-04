#!/bin/bash
# experiment.sh
# =============
# Convenience script to run all sparsity × client configurations
# from the Lightweight-Fed-NIDS paper (Bouayad et al., 2024).
#
# Paper Table II & III: sparsity=[0%,50%,70%,90%] × clients=[10,50,100]
#
# Usage:
#   chmod +x experiment.sh
#   bash experiment.sh
#   bash experiment.sh --partition non_iid

PARTITION=${1:-iid}
RAW_DATA="data/USTC-TFC2016"
PROC_DATA="data/processed"
IMAGE_SIZE=224
ROUNDS=5
LOCAL_EPOCHS=10
LR=0.0002
SEED=42

echo "========================================================"
echo " Lightweight-Fed-NIDS Full Experiment"
echo " Dataset   : USTC-TFC2016"
echo " Model     : ResNet-50"
echo " Partition : $PARTITION"
echo "========================================================"

for CLIENTS in 10 50 100; do
  for SPARSITY in 0.0 0.5 0.7 0.9; do
    echo ""
    echo "--------------------------------------------------------"
    echo " Clients=${CLIENTS}, Sparsity=${SPARSITY}"
    echo "--------------------------------------------------------"
    python main.py \
      --raw_data     "$RAW_DATA" \
      --proc_data    "$PROC_DATA" \
      --image_size   $IMAGE_SIZE \
      --num_clients  $CLIENTS \
      --sparsity     $SPARSITY \
      --rounds       $ROUNDS \
      --local_epochs $LOCAL_EPOCHS \
      --lr           $LR \
      --partition    $PARTITION \
      --seed         $SEED
  done
done

echo ""
echo "========================================================"
echo " All experiments complete. Results in EXPERIMENT/ folder"
echo "========================================================"
