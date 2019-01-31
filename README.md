# ImageProcessing
This is a Cuda Based PGM image manipulator that includes Edge detection and contour mapping. 
To test on a linux machine simply give permission to execute to the .sh file with the input pictures and output picture file. 


To run 

  run makefile
  
  ./project -(command) -[OPTIONAL COMMANDS] inputPicture.ascii.pgm outputPicture.ascii.pgm


-e = draw black edge around photo

-c = draw circle give a radius

-sy = synthesize white pgm file of given size

-d = edge detection given a threshold

-g = gaussian blur given a blur radius

-z = zhang suen thinning algorithm

-h = guo hall thinning algorithm

-m = median filter

-C = contour map

-l = draw line given two endpoints
