#! /bin/bash

cd build
make
mkdir frames
./StreamFilterApp
ffmpeg -framerate 15 -pattern_type glob -i 'frames/*.jpg' -c:v libx264 out.mp4
rm -rf frames
