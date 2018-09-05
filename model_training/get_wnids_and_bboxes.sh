# n04451818 is wnid of 'tool'
HYPONYM_URL="http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid=n04451818&full=1"
BBOX_URL="http://image-net.org/api/text/imagenet.bbox.obtain_synset_wordlist"

HYPONYM="n04451818_hyponym.txt"
BBOX="bbox_classes.txt"

# download hyponym and bounding boxes
if [ ! -f "$HYPONYM" ]; then
	curl -s -L ${HYPONYM_URL} > "$HYPONYM"
	sed -i -e 's/-//g' $HYPONYM
fi
if [ ! -f "$BBOX" ]; then
	curl -s -L ${BBOX_URL} > "$BBOX"
fi

# http://www.image-net.org/api/text/imagenet.synset.geturls.getmapping?wnid=[wnid]
IMAGE_URL="http://www.image-net.org/api/text/imagenet.synset.geturls.getmapping?wnid="
# http://www.image-net.org/api/download/imagenet.bbox.synset?wnid=[wnid]
IMAGE_BBOX_URL="http://www.image-net.org/api/download/imagenet.bbox.synset?wnid="

# download image urls and bounding boxes
if [ ! -d "www.image-net.org" ]; then
	while read line; do
		WNID=${line//[$'\t\r\n ']}
		if grep -q "$WNID" "$BBOX"; then
			wget -r "$IMAGE_URL$WNID"
			wget -r "$IMAGE_BBOX_URL$WNID"
		fi
	done < "$HYPONYM"
fi

# unzip the downloaded files
if [ ! -d "Annotation" ]; then
	mkdir "Annotation"
	for tar in www.image-net.org/downloads/bbox/bbox/*.tar.gz; do
		tar -xf $tar
	done
fi

# move image urls
if [ ! -d "image_urls" ]; then
	mkdir "image_urls"
	for file in www.image-net.org/api/text/*; do
		FILE_NAME=$(echo $file | cut -d'?' -f2 | cut -d'=' -f2)
		cp $file image_urls/$FILE_NAME
	done
fi