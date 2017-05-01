#!/bin/bash

echo -n "Please write an output filename (example: output.tar.gz): " 
read filename

echo -n "Please write the prefix for the file to be zipped (optional): "
read prefix

tar -cvf $filename ${prefix}*.png cost_trace.txt
rm ${prefix}*.png cost_trace.txt
echo "tar successfully filename is " >> $filename
