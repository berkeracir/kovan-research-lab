#!/bin/bash
if [ ! -d "Images" ]; then
	mkdir "Images"
fi

for file in Annotation/*; do
	WNID=$(echo $file | cut -d'/' -f2)
	if [ ! -d "Images/$WNID" ]; then
		mkdir "Images/$WNID"
	fi

	for subfile in "$file"/*; do
		IMAGE=$(echo $subfile | cut -d'/' -f3 | cut -d'.' -f1)
		IMAGE_URL=$(grep "$IMAGE" image_urls/$WNID | cut -d' ' -f2)
		IMAGE_URL=${IMAGE_URL//[$'\t\r\n ']}
		if [ ! -f "Images/"$WNID"/$IMAGE.${IMAGE_URL##*.}" ]; then
			curl --max-time 2 --connect-timeout 2 -o "Images/"$WNID"/$IMAGE.${IMAGE_URL##*.}" "$IMAGE_URL"
		fi
	done
done

# remove undisplayable images
for file in Images/*; do
	for image in "$file"/*; do 
		identify -quiet $image && : || rm $image
	done
done