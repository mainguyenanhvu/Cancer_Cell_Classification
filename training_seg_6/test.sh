#!/bin/bash
moveFile() {
	while read data; do 
		echo "[F:$(date +"%D %T")] $data $1"; 
		mkdir -p $1
    	mv $data $1
	done;
}

cat list.csv |
while IFS=',' read -ra ADDR; do
	echo ${ADDR[1]}
    find . -type f -name "${ADDR[0]}*" | moveFile ${ADDR[1]}
done