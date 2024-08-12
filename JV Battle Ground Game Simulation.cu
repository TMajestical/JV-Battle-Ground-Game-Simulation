#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>
#include<thrust/device_ptr.h>
#include <thrust/reduce.h>

using namespace std;

//*******************************************

// Write down the kernels here


__global__ void battle_round(int *X,int * Y,int *alive,int *score,int *HP,int round,int T)
{

  if(round%T) //Checking if it is a non null round
  {
    __shared__ unsigned long long int closest_distance;

    
      if(threadIdx.x == 0)
          closest_distance = ULLONG_MAX; // For each tank (thread block). maintain the closest tank along the line of fire

      __syncthreads();

      int firing_tank = blockIdx.x;
      int target_tank = (firing_tank+round)%T; //computing target tank as specifed in PS.

      int current_tank = threadIdx.x; //tank handled by the thread

      if(alive[firing_tank] && alive[current_tank] && firing_tank != current_tank) //process further iff tank and current thread is not processing the firing tank
      {
          //find the line of firing vector as x and y components
          long long firing_vector_x = X[target_tank] - X[firing_tank];
          long long firing_vector_y = Y[target_tank] - Y[firing_tank];

          //find the vector of current tank as x and y components
          long long cur_tank_vector_x = X[current_tank] - X[firing_tank];
          long long cur_tank_vector_y = Y[current_tank] - Y[firing_tank];

          // if current tank vector is in line of firing and it happens to be the closest tank alive in the direction
          // then that tank gets hit.

          /*
          Find if the cur_tank is in the line of firing:

            Idea is based on slope : (y2-y1)/(x2-x1).

            Considering 3 points firing tank, target tank and current tank.

            slope of line of firing (m1) : firing_vector_y/firing_vector_x

            slope between firing tank and current tank (m2) : cur_tank_vector_y/cur_tank_vector_x

            conditions for firing tank, target tank and current tank to fall on the same line:

              1. If they are aligned along x axis or y axis, then the slopes are 0 and inf respectively.

                ## In the case where they all are parallel to x-axis : slope zero => firing_vector_y = cur_tank_vector_y = 0, then yes current tank is in line of fire.
                ## In the case where they all are parallel to y-axis : slope is infinity => firing_vector_x = cur_tank_vector_x = 0, then yes current tank is in line of fire.

              2. The lines are not parallel to x or y axes, if slope of firing vector == slope of current tank and direction of firing is also same, then current tank gets hit
          */



          if(((firing_vector_x == cur_tank_vector_x && cur_tank_vector_x == 0) && (firing_vector_y*cur_tank_vector_y>0)) || ((firing_vector_y == cur_tank_vector_y && cur_tank_vector_y == 0) && (firing_vector_x*cur_tank_vector_x>0)) || ((firing_vector_y*cur_tank_vector_x == firing_vector_x*cur_tank_vector_y) && (firing_vector_x*cur_tank_vector_x>0) && (firing_vector_y*cur_tank_vector_y>0)))
          {
              // now that the cur_tank is in the line of firing the job is to find if it is the closest tank in this line so far.
              // If yes then that will be the tank that is hit

              long long int distance = cur_tank_vector_x*cur_tank_vector_x + cur_tank_vector_y*cur_tank_vector_y; //just the standadard eucledian distance without square root.

              atomicMin(&closest_distance,distance);

              __syncthreads(); //TRY :  without this later.

              if(distance == closest_distance) //if the tank is the closest, then it takes the hit.
              {
                  score[firing_tank]++; //need not be atomic, because score of each tank is updated by atmost 1 thread.
                  atomicSub(&HP[current_tank],1);
              }
          }
      }
  }

}

__global__ void is_end_game(int * HP,int * alive)
{
    /*
    The task of the kernel is to identify the tanks that are alive

    If atleast 2 tanks are alive, there would be another round, else the game ends there.
    */

    int id = threadIdx.x;

    if(HP[id]<=0)
        alive[id] = 0;

}

/*
For some reason memset to nonzero values is filling up with garbage values
Hence using this kernel as a substitute.

*/
__global__ void intialize_array(int *A, int value ,int n)
{

  A[threadIdx.x] = value;
}

//***********************************************


int main(int argc,char **argv)
{
    // Variable declarations
    int M,N,T,H,*xcoord,*ycoord,*score;


    FILE *inputfilepointer;

    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0;
    }

    fscanf( inputfilepointer, "%d", &M );
    fscanf( inputfilepointer, "%d", &N );
    fscanf( inputfilepointer, "%d", &T ); // T is number of Tanks
    fscanf( inputfilepointer, "%d", &H ); // H is the starting Health point of each Tank

    // Allocate memory on CPU
    xcoord=(int*)malloc(T * sizeof (int));  // X coordinate of each tank
    ycoord=(int*)malloc(T * sizeof (int));  // Y coordinate of each tank
    score=(int*)malloc(T * sizeof (int));  // Score of each tank (ensure that at the end you have copied back the score calculations on the GPU back to this allocation)

    // Get the Input of Tank coordinates
    for(int i=0;i<T;i++)
    {
      fscanf( inputfilepointer, "%d", &xcoord[i] );
      fscanf( inputfilepointer, "%d", &ycoord[i] );
    }


    auto start = chrono::high_resolution_clock::now();

    //*********************************
    // Your Code begins here (Do not change anything in main() above this comment)
    //********************************

    //creating streams to do mem copies and allocs async.ly
    cudaStream_t stream1,stream2,stream3;//,stream4;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    //cudaStreamCreate(&stream4);

    //defining pointers for device arrays
    int *d_xcoord,*d_ycoord,*d_score,*d_HP,*d_alive; //d_alive is used to maintain the list of tanks that are alive.

    //allocating mem for device arrays
    cudaMalloc(&d_xcoord,T*sizeof(int));
    cudaMalloc(&d_ycoord,T*sizeof(int));
    cudaMalloc(&d_score,T*sizeof(int));
    cudaMalloc(&d_HP,T*sizeof(int));
    cudaMalloc(&d_alive,T*sizeof(int));

    //Memcpy/set Async

    intialize_array<<<1,T,0,stream1>>>(d_alive,1,T);
    cudaMemcpyAsync(d_xcoord,xcoord,T*sizeof(int),cudaMemcpyHostToDevice,stream1);

    intialize_array<<<1,T,0,stream2>>>(d_HP,H,T);
    cudaMemcpyAsync(d_ycoord,ycoord,T*sizeof(int),cudaMemcpyHostToDevice,stream2);

    intialize_array<<<1,T,0,stream3>>>(d_score,0,T);


    //wait for above operations to complete.
    cudaDeviceSynchronize();


    /*
    Approach:

        Launch a kernel for each, round, such that:

            Each tank firing is handled by a thread block with T threads. (T : number of tanks).

            Job of a threadblock is to see all the T tanks in parallel and identify the tank that gets hit (if any).

            vector formed by two points (the firing tank and the targeted tank) gives:
                Direction of firing (slope), and distance between two points.

            Job of each thread is to check if it's tank is in the line of fire and is the closest (first to get impacted)
            And accordingly update it.

            Later on the threadblock decreases HP of the affected tank and increases the score of the firing tank.
    */

    int round = 1; // Round 0, is a null round anyways.
    int game_on = 1;

    //given in PS that 2<=T<=1000
    int blocks = T; // 1 thread block per tank
    int block_size = T; //1 thread per tank in a threadblock


    thrust::device_ptr<int> thrust_alive(d_alive); //wrap device pointer with trust to use thrust algos

    while(game_on) //loop that stops when the game is over.
    {
        
        //To increase the amount of computation as compared to other unuseful operations
        for(int jj =0; jj < 10; jj++)
        {
          // Simulate a battle round using the following kernel call
          battle_round<<<blocks,block_size>>>(d_xcoord,d_ycoord,d_alive,d_score,d_HP,round,T);
          // Based on the HP alive status of each tank is updated by the below kernel.
          is_end_game<<<1,T>>>(d_HP,d_alive);

          round++;
        }
        
        cudaDeviceSynchronize();

        int alive_tanks = thrust::reduce(thrust_alive,thrust_alive+T);

        if (alive_tanks <= 1)
          {
              game_on = 0; // the condition that determines if the outer loop should proceed.
              break;
          }
      }


    

    cudaMemcpy(score,d_score,T*sizeof(int),cudaMemcpyDeviceToHost); //copy final scores of each tank to the host.

    //*********************************
    // Your Code ends here (Do not change anything in main() below this comment)
    //********************************

    auto end  = chrono::high_resolution_clock::now();

    chrono::duration<double, std::micro> timeTaken = end-start;

    printf("Execution time : %f\n", timeTaken.count());

    // Output
    char *outputfilename = argv[2];
    char *exectimefilename = argv[3];
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    for(int i=0;i<T;i++)
    {
        fprintf( outputfilepointer, "%d\n", score[i]);
    }

    fclose(inputfilepointer);
    fclose(outputfilepointer);

    outputfilepointer = fopen(exectimefilename,"w");
    fprintf(outputfilepointer,"%f", timeTaken.count());
    fclose(outputfilepointer);

    free(xcoord);
    free(ycoord);
    free(score);
    cudaDeviceSynchronize();
    return 0;
}