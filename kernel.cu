#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include "definitions.h"

// Cuda Header
cudaError_t calculateDistanceCuda(Point *pointArray, int sizeOfPointArray,
Distance *distanceArray, int sizeOfDistanceArray,int arrayPointer,int num);

__global__ void calculate(Point *pointArray, int sizeOfPointArray,
Distance *distanceArray, int sizeOfDistanceArray,int block,int arrayPointer,int num)
{
for (int i = 0; i < num ; i++)
{
int myThreadId = threadIdx.x + blockIdx.x * block; //my index in all GPU
float tempx,tempy,res;

float myx = pointArray[i + arrayPointer].x;
float myy = pointArray[i + arrayPointer].y;
res = powf(abs(myx-pointArray[myThreadId].x),2) + powf(abs(myy-pointArray[myThreadId].y),2);
distanceArray[myThreadId+i*sizeOfPointArray].id = myThreadId;
distanceArray[myThreadId+i*sizeOfPointArray].distance = sqrt((float)res);
}
}

// The main method that run in main.ccp
cudaError_t calculateDistanceCuda(Point *pointArray, int sizeOfPointArray,
Distance *distanceArray, int sizeOfDistanceArray, int arrayPointer,int num)
{
cudaError_t cudaStatus;
Point* dev_pointArray = 0;
Distance*  dev_distanceArray = 0;

cudaStatus = cudaMalloc((void**)&dev_pointArray, sizeOfPointArray*sizeof(Point));

cudaStatus = cudaMalloc((void**)&dev_distanceArray, sizeOfDistanceArray*sizeof(Distance));

// Copy pointArray array from host memory to GPU buffers.
cudaStatus = cudaMemcpy(dev_pointArray,pointArray, sizeOfPointArray * sizeof(Point), cudaMemcpyHostToDevice);

int blocks = sizeOfPointArray/NUMOFTHREADS;

// Launch a kernel on the GPU
calculate<<< blocks, NUMOFTHREADS >>>(dev_pointArray, sizeOfPointArray, dev_distanceArray, sizeOfDistanceArray, NUMOFTHREADS,arrayPointer ,num);

cudaStatus = cudaDeviceSynchronize();

// Copy dev_distanceArray  from GPU buffer to host memory.
cudaStatus = cudaMemcpy(distanceArray, dev_distanceArray, sizeOfDistanceArray*sizeof(Distance), cudaMemcpyDeviceToHost);

return cudaStatus;
}
