#!/bin/bash

echo -n "Please write an output filename (example: output.tar.gz): " 
read filename
tar -cvf $filename *.jpg *.png
rm *.jpg *.png
echo "tar successfully filename is " >> $filename
