#!/bin/bash

RCI_DIR="/mnt/personal/korcadav/eforce_cone_seg/train_results/*"
DEST_DIR="./rci_results"
mkdir -p "$DEST_DIR"
rsync -avz --progress --ignore-existing korcadav@login3.rci.cvut.cz:"$RCI_DIR" "$DEST_DIR"
