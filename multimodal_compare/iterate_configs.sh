#!/bin/bash
FILES=$@
COUNTER=0
cd ~/multimodal-vae-comparison/multimodal_compare
for f in $FILES
do
  COUNTER=$(( COUNTER + 1 ))
  echo "$f"
  python main.py --cfg "$f"
done
