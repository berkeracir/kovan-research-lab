INPUT=$1

BBOX_CLASSES="bbox_classes.txt"
URL="http://image-net.org/api/text/imagenet.bbox.obtain_synset_wordlist"
if [ ! -f "$BBOX_CLASSES" ]; then
    curl -s -L ${URL} | cut -d'>' -f2 | cut -d'<' -f1 > "$BBOX_CLASSSES"
fi

if [ -f "$INPUT" ]; then
    while read line; do
        item=$(echo "$line" | cut -d'(' -f1 | cut -d' ' -f1)
        if grep -q -s "^$item$" "$BBOX_CLASSES"; then
            echo $item
        fi
    done < "$INPUT"
fi
