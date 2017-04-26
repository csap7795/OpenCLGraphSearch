#!/bin/bash

_files=csv_files/*.csv
for f in $_files;
do
	fbname=$(basename "$f" .csv)
	pdflatex --jobname=$fbname "\def\csvFile{$f} \input{dynamic.tex}"
done

rm *.aux
rm *.log