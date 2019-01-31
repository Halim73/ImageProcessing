#!/bin/bash
printf "starting line detections"
./project -d 20 ./inputPictures/baboon.ascii.pgm baboonOut.ascii.pgm
./project -d 5 ./inputPictures/balloons.ascii.pgm balloonOut.ascii.pgm
./project -d 12 ./inputPictures/brain.ascii.pgm brainOut.ascii.pgm
./project -d 12 ./inputPictures/lena.ascii.pgm lenaOut.ascii.pgm
./project -d 15 ./inputPictures/casablanca.ascii.pgm casaOut.ascii.pgm
./project -d 15 ./inputPictures/f18.ascii.pgm f18Out.ascii.pgm
./project -d 3 ./inputPictures/frog.ascii.pgm frogOut.ascii.pgm
./project -d 10 ./inputPictures/eagle.ascii.pgm eagleOut.ascii.pgm
./project -d 20 ./inputPictures/brainScan.ascii.pgm brainScanOut.ascii.pgm
./project -d 10 ./inputPictures/SatImage1.ascii.pgm satOut.ascii.pgm
mv *.pgm outputPictures
echo "finished processing images"
