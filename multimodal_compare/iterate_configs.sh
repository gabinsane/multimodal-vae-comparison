#!/bin/bash
FILES=$@
COUNTER=0
cd ~/multimodal-vae-comparison/multimodal_compare
allfiles=$(find $FILES -name "*.yml")
for f in $allfiles;
do
  COUNTER=$(( COUNTER + 1 ));
  echo "Config $f";
  python main.py --cfg "$f";
done
echo "Finished all $COUNTER experiments"
