#!/bin/bash
export LD_LIBRARY_PATH="$PWD/bin/Debug"

_files=Graph/*.g
for f in $_files;
do
	_filename=$(basename $_files)
	./myprog "$f" #> logs/"$_filename.txt"
done
