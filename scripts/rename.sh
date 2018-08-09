#!/bin/bash

ID=0
for file in *.jp*g; do 
	echo $file "->" test_image_$ID.${file##*.}
	mv $file test_image_$ID.${file##*.}
	ID=$((ID+1))
done
