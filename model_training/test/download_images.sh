#!/bin/bash
for file in Annotation/n04154565/*; do
    IMAGE=$(echo $file | cut -d'/' -f3 | cut -d'.' -f1)
    IMAGE_URL=$(grep "$IMAGE" imagenet.synset.geturls.txt | cut -d' ' -f2)
    IMAGE_URL=${IMAGE_URL//[$'\t\r\n ']}
    curl -o "Images/n04154565/$IMAGE.${IMAGE_URL##*.}" "$IMAGE_URL"
done

# remove undisplayable images
for image in Images/n04154565/*; do 
    identify -quiet $image && : || mv $image Images/
done
