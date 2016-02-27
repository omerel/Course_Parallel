// Scruct of Cords for a point
typedef struct{
    float x;
    float y;
}Point;

// Struct of Distance
typedef struct{
    int id;
    float distance;
}Distance;

#define INPUT "//LIBRARY-12/Debug/input.txt"	// File path input.txt
#define OUTPUT "//LIBRARY-12/Debug/output.txt"	// File path input.txt
#define MASTER 0			// Define rank == 0
#define NUM 100				// Size of Distance array to send to CUDA in one command. size of points must be divided in NUM.
#define NUMOFTHREADS 1000	// Define how many threads we want to use in each Block
#define RATIO 30
