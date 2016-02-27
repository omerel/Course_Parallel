#include "cuda_runtime.h"
#include <stdio.h>
#include<stdlib.h>
#include <mpi.h>    // use for MPI
#include <omp.h>    // use for OpenMP
#include <ctime>    //
#include "definitions.h"
#include <math.h>


//////////////////////////////////////////////////////////////////////////////////////////////////
////////																				//////////
////////                     Omer Elgrably 21590807   final project                     //////////
////////																				//////////
//////////////////////////////////////////////////////////////////////////////////////////////////


// Cuda Header
cudaError_t calculateDistanceCuda(Point *pointArray, int sizeOfPointArray, Distance *distanceArray, int sizeOfDistanceArray,int arrayPointer, int num);

// calculateDistanceSeires
void calculateDistanceSeires(Point *pointArray, int sizeOfPointArray, Distance *distanceArray, int sizeOfDistanceArray,int arrayPointer, int num)
{
	int myId;
	float myx,myy,tempx,tempy,res;
	for (int i = 0 ; i < num ; i++)
	{
		myx = pointArray[arrayPointer + i].x;
		myy = pointArray[arrayPointer + i].y;
		for (int j = 0; j < sizeOfPointArray; j++)
		{
			distanceArray[sizeOfPointArray*i+j].id = j;
			res =  powf(abs(myx-pointArray[j].x),2) + powf(abs(myy-pointArray[j].y),2);
			distanceArray[sizeOfPointArray*i+j].distance = sqrt((float)res);
		}
	}
}
// Swap method  - using for sort method
void swap(Distance *a, Distance *b)
{
    Distance t;
    t = *a;
    *a = *b;
    *b = t;
}

// K_sort method - sort the first k samllest nembers from the given array
void k_sort(Distance* distanceArray, int size , int k )
{
    for (int i = 0 ; i < k+1; i++)
        for (int j = i ; j < size; j++)
            if (distanceArray[i].distance > distanceArray[j].distance)
                swap(&distanceArray[i],&distanceArray[j]);
}

// Method that read the points from the given file and put them in an array
// read: num of points,K elemnts and after it all the data.
Point*  getCoords(char* fileName, int* nSize, int* k)
{
    int id;
    FILE* f = fopen(fileName,"r");
    fscanf(f,"%d %d",nSize,k);
    Point* pointArray = (Point*)calloc(*nSize, sizeof(Point));
    for (int i = 0; i < *nSize ; i++)
    {
        fscanf(f,"%d %f %f",&id,&pointArray[i].x,&pointArray[i].y);
    }
    fclose(f);
    return  pointArray;
}

// Method that print the result to file and screen.
void writeResult(int num,FILE* f,Distance* printArray,int kElemnts)
{
	for (int i = 0; i < num; i++)
	{
		fprintf(f,"%d : ",printArray[(kElemnts+1)*i].id);		// Print Point id to file
		for(int l = 1; l <= kElemnts; l++)						// Print K closest id
			fprintf(f,"%d, ",printArray[l+i*(kElemnts+1)].id);  // Print Point id to Screen
		fprintf(f,"\n\n");
	}
}

// Method that check cudaSuccess
void cudaCheck(cudaError_t cudaStatus)
{
	if (cudaStatus != cudaSuccess)			// Check cudaSuccess and reset
		printf("----------CUDA ERROR!----------\n");
	cudaStatus = cudaDeviceReset();
}

// Method print finish and time
void printFinish(FILE* f,int nSize,clock_t t_end,clock_t t_start)
{
	fprintf(f, "--------------------------------Finish-------------------------------------\n");
	fprintf(f, "| %d Points | NUM : %d | CUDA Threads: 1000 | CUDA Blocks : %d |\n",nSize,NUM,nSize/1000);
	fprintf(f, "----------------------------------------------------------------------------\n");
    fprintf(f, "Total time : %.4f seconds.\n",(float)(t_end - t_start)/CLOCKS_PER_SEC);
}

// Method that copy to printarray the k closest point from distanceArray
void createPrintArray(Distance* printArray, Distance* distanceArray,int i,int kElemnts,int nSize)
{
	for (int j = 0; j <= kElemnts ; j++)
		{
			printArray[i*(kElemnts+1) + j].id = distanceArray[i*nSize + j].id;
			printArray[i*(kElemnts+1) + j].distance = distanceArray[i*nSize + j].distance;
		}
}

void sortAndCreatePrintArray(int num,Distance* distanceArray,int nSize,int kElemnts,Distance* printArray)
{
	for(int i = 0; i < num; i++)
	{
		k_sort(&distanceArray[i*nSize], nSize ,kElemnts);
		createPrintArray(printArray,distanceArray,i,kElemnts,nSize);
	}
}

void initilaize_startPointer_and_counter(int rank,int numprocs,int nSize,int*pointer,int*counter)
{
	if (rank == MASTER)
	{
		*pointer = 0;
		*counter = (nSize*RATIO/100);
	}
	else
	{
		*counter = ((nSize*(100-RATIO)/100)/(numprocs-1));
		*pointer = (nSize*RATIO/100) + (*counter)*(rank-1);
	}
}

//Main
int main(int argc, char *argv[])
{
    // MPI initalize
    int numprocs, rank;
    MPI_Status status;
    
	MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Create MPI_DISTANCE 
    Distance distance;
    MPI_Datatype MPI_DISTANCE;
    MPI_Datatype type[2] = { MPI_INT, MPI_FLOAT };
    int blocklen[2] = { 1, 1 };
    MPI_Aint disp[2];

    disp[0] = (char *) &distance.id - (char *) &distance;
    disp[1] = (char *) &distance.distance - (char *) &distance;
    MPI_Type_create_struct(2, blocklen, disp, type, &MPI_DISTANCE);
    MPI_Type_commit(&MPI_DISTANCE);

    
    clock_t t_start,t_end;			// Define time stamps.
    int kElemnts;                   // first K elements to print.
    int nSize;                      // Size of N points.
    Point* pointArray;              // Array of all the points from the given file.
    Distance* distanceArray;        // Array which contain for each point the distance to each point in the array.
	Distance* printArray;			// Array which save the first closest k points. will be printed in the end of th algoritem.
    int counter;                    // Loop counter if distanceArray is divided to more then one array (NUM > 1).
	int finish = 0;					// process sent this value when it finish work.
	int pointer;					// pointer for pointArray- where to start calculate distance.

    if (rank == MASTER)
    {
        t_start = clock();										// Start measure time.									
        pointArray = getCoords(INPUT,&nSize,&kElemnts);			// Read points from file to pointArray.
	}

    MPI_Bcast(&nSize,1,MPI_INT,MASTER,MPI_COMM_WORLD);			// Update master's nSize to all.
	MPI_Bcast(&kElemnts,1,MPI_INT,MASTER,MPI_COMM_WORLD);		// Update master's kElemnts to all.
	initilaize_startPointer_and_counter(rank,numprocs,nSize,&pointer,&counter);
	
	// broadcast to slave point array- didn't secceed to put it in method
	if (rank != MASTER)
		pointArray = (Point*)calloc(nSize, sizeof(Point));		// initalaize pointArray in slaves.
	for (int i = 0; i < nSize; i++)								// broadcast x and y.
		{
			MPI_Bcast(&pointArray[i].x,1,MPI_FLOAT,MASTER,MPI_COMM_WORLD);
			MPI_Bcast(&pointArray[i].y,1,MPI_FLOAT,MASTER,MPI_COMM_WORLD);
		}

	counter = counter/NUM;										// devide the counter for the For Loop.

    if (rank == MASTER)											// IF MASTER
    {
		FILE* f;
#pragma omp parallel shared(pointer, nSize, counter, pointArray, finish,printArray,f)
        {
#pragma omp sections
            {
#pragma omp section												// MASTER'S section 1- calculate distance with GPU and sort
                {
                    distanceArray = (Distance*)calloc(nSize*NUM,sizeof(Distance));
					Distance* tempPrintArray =  (Distance*)calloc(NUM*(kElemnts+1),sizeof(Distance));

                    for (int i = 0; i < counter; i++)			//
                    {											//calculateDistance(pointArray, size, distanceArray, size,pointer on distanceArray ,size of NUM);
						cudaError_t cudaStatus = calculateDistanceCuda(pointArray, nSize, distanceArray, nSize*NUM,i*NUM,NUM);
						cudaCheck(cudaStatus);					// chcek cudaSucsses method
						sortAndCreatePrintArray(NUM,distanceArray,nSize,kElemnts,tempPrintArray);
						MPI_Send(tempPrintArray, NUM*(kElemnts+1), MPI_DISTANCE, MASTER,0, MPI_COMM_WORLD);
                    }
					free(tempPrintArray);
                    free(distanceArray);
                }
#pragma omp section												// MASTER'S section 2 - get the printArrays from all processes
                {
					int index = 0;
					printArray =  (Distance*)calloc(nSize*(kElemnts+1),sizeof(Distance));
					for (int i = 0; i < nSize/NUM; i++)
					{
						MPI_Recv(&printArray[index], NUM*(kElemnts+1), MPI_DISTANCE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
						index += NUM*(kElemnts+1);
					}
					MPI_Send(&finish, 1, MPI_INT, MASTER,1, MPI_COMM_WORLD);
				}

#pragma omp section												// MASTER'S section 3 - print and finalize
                {
					for (int i = 0; i < numprocs ; i++)
						MPI_Recv(&finish, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);

                    FILE* f = fopen(OUTPUT, "w+");				// Create ouput file
					writeResult(nSize,f,printArray,kElemnts);

                    free(printArray);							// free memory
                    t_end = clock();							// stop mesure time
					printFinish(f,nSize,t_end,t_start);			// print finish method
					fclose(f);                                  // close file;
					printf("Work finished. check file.\n");
					MPI_Finalize();
                }
            }
        }
    }

    else // If SLAVES
	{
		int i = 0;
#pragma omp parallel shared(nSize, pointArray,pointer,counter) private(i,distanceArray,printArray)
        {
			distanceArray = (Distance*)calloc(nSize*NUM,sizeof(Distance));
			printArray =  (Distance*)calloc(NUM*(kElemnts+1),sizeof(Distance));
#pragma omp for
			for (i = 0 ; i < counter; i++ )			//
			{	
				calculateDistanceSeires(pointArray, nSize, distanceArray, nSize*NUM,pointer + NUM*i,NUM);
				sortAndCreatePrintArray(NUM,distanceArray,nSize,kElemnts,printArray);
				MPI_Send(printArray, NUM*(kElemnts+1), MPI_DISTANCE, MASTER,0, MPI_COMM_WORLD);
			}
			free(printArray);
			free(distanceArray);
		}
		MPI_Send(&finish, 1, MPI_INT, MASTER,1, MPI_COMM_WORLD);
		MPI_Finalize();
	}

}
