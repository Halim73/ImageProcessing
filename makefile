project: pgmProcess.o pgmUtility.o main.o
	nvcc -arch=sm_30 -o project pgmProcess.o pgmUtility.o main.o

pgmProcess.o: pgmProcess.cu pgmProcess.h
	nvcc -arch=sm_30 -c pgmProcess.cu
		   
pgmUtility.o: pgmUtility.cu pgmUtility.h
	nvcc -arch=sm_30 -c pgmUtility.cu

main.o: main.c main.h 
	g++ -c -x c++ main.c -I -lcudart -lcuda.

clean:
	rm main.o
	rm pgmUtility.o
	rm pgmProcess.o
