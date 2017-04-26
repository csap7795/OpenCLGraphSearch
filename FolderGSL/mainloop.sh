#!/bin/bash
export LD_LIBRARY_PATH="$PWD/bin/Debug"

_files=Graph/*.g
for f in $_files;
do
	./myprog "$f"
	echo $f
done
