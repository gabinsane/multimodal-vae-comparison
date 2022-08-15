#!/bin/bash
FILES=$@
cd ~/multimodal-vae-comparison/multimodal_compare
allfiles= find $FILES -name "*.yml"
for f in $allfiles;
  do echo "Config $f";
  python main.py --cfg "$f"
done
