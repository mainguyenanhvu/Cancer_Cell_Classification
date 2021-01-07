#!/bin/bash
find . -type f | xargs -I _ dirname _ | sort | uniq -c | while read n d; do
  echo "=== $d ($n files) ===";
  if [ $(($n / $FRACTION)) -gt 0 ]; then
    find "$d" -type f | sort -R --random-source=<(python rand_bits.py 5) | head -n $(($n / $FRACTION)) | while read file; do
      echo "$file  ->  $NEW_DIR/$d";
      mkdir -p "$NEW_DIR/$d";
      mv "$file" "$NEW_DIR/$d";
    done;
  fi;
  echo;
done