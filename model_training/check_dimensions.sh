#!/bin/bash
for wnid_path in Images/*; do
    wnid=$(echo $wnid_path | cut -d'/' -f2)
    for image_path in "$wnid_path"/*; do
        image=$(echo $image_path | cut -d'/' -f3 | cut -d'.' -f1)
        width=$(identify -format "%w" "$image_path")> /dev/null
        xml_width=$(grep "width" Annotation/"$wnid"/"$image".xml | cut -d'>' -f2 | cut -d'<' -f1)
        height=$(identify -format "%h" "$image_path")> /dev/null
        xml_height=$(grep "height" Annotation/"$wnid"/"$image".xml | cut -d'>' -f2 | cut -d'<' -f1)
        
        if [ ! "$width" -eq "$xml_width" ] || [ ! "$height" -eq "$xml_height" ]; then
            echo "Shape mismatch in $image with ($width,$xml_width) ($height,$xml_height)"
            rm -v $image_path
        fi
    done
done