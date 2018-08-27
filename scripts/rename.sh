ID=1
while read line; do
	if [[ $line != "!"* ]]; then
		string="$(grep -B 3 -A 1 "$line" ~/kovan-research-lab/object_detection/data/oid_bbox_trainable_label_map.pbtxt)"
		if [[ $string = "item"* ]]; then
			echo "${string/id: ???/id: $ID}"
			ID=$((ID+1))
		fi
	fi
done < ~/kovan-research-lab/datasets/Open\ Images\ Dataset\ V4.txt
