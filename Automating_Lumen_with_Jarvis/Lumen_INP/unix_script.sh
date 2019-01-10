#!/bin/bash
awk '{ sub("\r$", ""); print }' runlumen > runlumen2 && mv runlumen2 runlumen
cd INP
for i in *; do
	awk '{ sub("\r$", ""); print }' $i > $i.2
	mv $i.2 $i
done
cd ../
