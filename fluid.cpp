#include <iostream>
#include<bits/stdc++.h>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

using namespace std;

struct node_t{ 
  double poten;
  double field; 
  double ix;
};

 struct particle_t{ 
  double q;              // charge
  double density; 
  double energy;
  double exH;
  double H;
  double flux;
  double M;               // mobility
  int z;                  // int charge
  double s;               // source or sink
  double su;              // source or sink of energy in elec[n]
  double D;               // diffusion
  double T;               // temperature
  double vth;             //Thermal velocity
 };

void meshing(node_t [], int, int, double, double,int,int,int);
double Potential_inside( int,node_t [], double , double ,double, int, int,int,int,int);
double Potential_borders( int,node_t [], double , double ,double, int, int,int,int,int,int);
void output(node_t [], int , int ,int);

int main(int argc, char *argv[])
{
 
  double h;           // [mm]
  double dt;
  double ArDensity;             // backgroumd gas density
  double P, eCharge, temp;      // background gas pressure
  double ep0;      
  double electemp;
  double emobility,Armobility; //mobility
  int n,m; 
  double n0;
  int mainIter, rStart=0;
  double volt, R, C, r;
 
  ifstream parameter;
  string line;  
  
  parameter.open ("parameter.txt");
  while(getline(parameter,line)) { 
    stringstream s(line);
    string command;
    s >> command;
    if(!command.compare("MAIN_IT"))            s >> mainIter;
    if(!command.compare("n0"))                 s >> n0;
    if(!command.compare("ep0"))                s >> ep0;       // x 10^-12
    if(!command.compare("eCharge"))            s >> eCharge;   // x 10^-19
    if(!command.compare("p"))                  s >> P;         // Pressure
    if(!command.compare("T"))                  s >> temp;      // Ar temperature  [eV]
    if(!command.compare("elecT"))              s >> electemp;  // ele temperature [ev]
    if(!command.compare("h"))                  s >> h ;        // [mm]
    if(!command.compare("dt"))                 s >> dt ;        // ns
    if(!command.compare("electronmobility"))   s >> emobility; // [mm^2/V/ns]
    if(!command.compare("Arpmobility"))        s >> Armobility;// [mm^2/V/ns]
    if(!command.compare("R"))                  s >> R ;        // [mm]
    if(!command.compare("Vs"))                 s >> volt ;        // [mm]
    if(!command.compare("C"))                  s >> C ;        // [mm]
    if(!command.compare("r"))                  s >> r ;        // [mm]
    if(!command.compare("rStart"))             s >> rStart ;        // restart from previous data
  }
  
  MPI_Init(&argc, &argv);
  clock_t start = clock(), diff; //turn on the clock and profiling
  MPI_Pcontrol(0);
  MPI_Barrier (MPI_COMM_WORLD);
  float t1 = MPI_Wtime();
  MPI_Pcontrol(1);
 
  //*********************************
  MPI_Datatype node_type;
  int lengths[3] = { 1, 1, 1 };
  MPI_Aint displacements[3];
  struct node_t dummy_node;
  MPI_Aint base_address;
  MPI_Get_address(&dummy_node, &base_address);
  MPI_Get_address(&dummy_node.poten, &displacements[0]);
  MPI_Get_address(&dummy_node.field, &displacements[1]);
  MPI_Get_address(&dummy_node.ix, &displacements[2]);
  displacements[0] = MPI_Aint_diff(displacements[0], base_address);
  displacements[1] = MPI_Aint_diff(displacements[1], base_address);
  displacements[2] = MPI_Aint_diff(displacements[2], base_address);
  MPI_Datatype types[3] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
  MPI_Type_create_struct(3, lengths, displacements, types, &node_type);
  MPI_Type_commit(&node_type);
  
  int comm_sz;  //number of processors
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  
  // Decompose a 2D cartesian grid
  MPI_Comm comm;
     
  int ndims = 2, reorder = 1;
  int periods[2] = {0, 0};
  int dims[2];
  dims[0] = 0;
  dims[1] = 0;
  MPI_Dims_create(comm_sz, 2, dims);

  // Make the cartesian topology
  MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &comm);
  enum DIRECTIONS {DOWN, UP, LEFT, RIGHT};
  const char *neighbours_names[4]={"down", "up", "left", "right"};
  int neighbours_ranks[4];
 
  // Let consider dims[0] = X, so the shift tells us our left and right neighbours
  MPI_Cart_shift(comm, 0, 1, &neighbours_ranks[LEFT], &neighbours_ranks[RIGHT]);
  // Let consider dims[1] = Y, so the shift tells us our up and down neighbours
  MPI_Cart_shift(comm, 1, 1, &neighbours_ranks[DOWN], &neighbours_ranks[UP]);
 
  // Get my rank in the new communicator
  int my_rank;
  MPI_Comm_rank(comm, &my_rank);
 
  for(int i = 0; i < comm_sz; i++) {
    if(neighbours_ranks[i] == MPI_PROC_NULL)
      printf("[MPI process %d] I have no %s neighbour.\n", my_rank, neighbours_names[i]);
    else
      printf("[MPI process %d] I have a %s neighbour: process %d.\n", my_rank, neighbours_names[i], neighbours_ranks[i]);
  }
  // Get my coordinates
  int my_coords[2];
  MPI_Cart_coords(comm, my_rank, ndims, my_coords);
  printf("[MPI process %d] Coords: (%d, %d).\n", my_rank, my_coords[0], my_coords[1]);
  
  //***************************
  n=m=10;                                     // n and m number of grids on x-axis amd y-azis
  int xBlockDimension, yBlockDimension;
  xBlockDimension = n / sqrt(comm_sz);        // grids for each subdomain on x-axis  
  yBlockDimension = m / sqrt(comm_sz);        // grids for each subdomain on y-axis 
        
  int maxXCount = xBlockDimension + 2;        // grids for each subdomain including the ghost celss
  int maxYCount = yBlockDimension + 2;        // grids for each subdomain including the ghost celss
 
  int local_x = my_coords[0] * xBlockDimension;
  int local_y = my_coords[1] * yBlockDimension;
  
  double xLeft = 0, xRight = 9;
  double yBottom = 0, yUp = 9;

  double deltaX = (xRight - xLeft) / (n - 1);
  double deltaY = (yUp - yBottom) / (m - 1);

#define TD2OD(XX,YY) ((YY)*maxXCount+XX)


  //create row and column 
  MPI_Datatype row;
  MPI_Type_contiguous(xBlockDimension, node_type, &row);
  MPI_Type_commit(&row);
  MPI_Datatype column;
  MPI_Type_vector(yBlockDimension, 1, maxXCount, node_type, &column);
  MPI_Type_commit(&column);
      
  node_t node[n*m];  
  
  meshing(node,local_x,local_y,deltaX,deltaY,xBlockDimension,yBlockDimension,my_rank);

  // main iteration <---------
  ////////////////////////////
  int iter = 0;
  double error_i, error_b, error=1, error_sum=1;
  MPI_Request rReqT, rReqB, rReqL, rReqR, sReqT, sReqB, sReqL, sReqR;

  while(iter++<mainIter) {
    // Receive
    MPI_Irecv(&node[TD2OD(1,maxYCount-1)].poten, 1, row, neighbours_ranks[UP], 0, comm, &rReqT);
    MPI_Irecv(&node[TD2OD(0,1)].poten, 1, column, neighbours_ranks[LEFT], 0, comm, &rReqL);
    MPI_Irecv(&node[TD2OD(1,0)].poten, 1, row, neighbours_ranks[DOWN], 0, comm, &rReqB);
    MPI_Irecv(&node[TD2OD(maxXCount-1,1)].poten, 1, column, neighbours_ranks[RIGHT], 0, comm, &rReqR);

    // send
    MPI_Isend(&node[TD2OD(1,1)].poten, 1, row, neighbours_ranks[DOWN], 0, comm, &sReqB);
    MPI_Isend(&node[TD2OD(maxXCount-2,1)].poten, 1, column, neighbours_ranks[RIGHT], 0, comm, &sReqR);
    MPI_Isend(&node[TD2OD(1,maxYCount-2)].poten, 1, row, neighbours_ranks[UP], 0, comm, &sReqT);
    MPI_Isend(&node[TD2OD(1,1)].poten, 1, column, neighbours_ranks[LEFT], 0, comm, &sReqL);
    
    //Wait for all
    MPI_Wait(&rReqT, MPI_STATUS_IGNORE);
    MPI_Wait(&rReqL, MPI_STATUS_IGNORE);
    MPI_Wait(&rReqB, MPI_STATUS_IGNORE);
    MPI_Wait(&rReqR, MPI_STATUS_IGNORE);
    
    MPI_Wait(&sReqB, MPI_STATUS_IGNORE);
    MPI_Wait(&sReqR, MPI_STATUS_IGNORE); 
    MPI_Wait(&sReqT, MPI_STATUS_IGNORE);
    MPI_Wait(&sReqL, MPI_STATUS_IGNORE);

    error_b = Potential_borders(maxXCount, node, ep0, deltaX,deltaY,  xBlockDimension,yBlockDimension, local_x,local_y,n,m);
    error_i = Potential_inside( maxXCount,node, ep0, deltaX,deltaY,  xBlockDimension,yBlockDimension, local_x,n,m);  
    
    error = error_i + error_b;
   
    if(iter%100==0 ) {
      cout << my_rank << " : " << error << " : " << iter << endl;  
      MPI_Reduce(&error, &error_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
      MPI_Bcast (&error_sum, 1, MPI_DOUBLE, 0, comm);
      
      if(error_sum<1.e-9) break;
    }
  }

  // write final results
  output(node, my_rank, xBlockDimension, yBlockDimension);

  MPI_Finalize();
  diff = clock() - start;
  if(my_rank == 0) {
    int msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("Time taken %d seconds %d milliseconds\n", msec / 1000, msec % 1000);
  }
 
  return 0;
}

void meshing(node_t node [], int local_x, int local_y, double deltaX, double deltaY,int xBlockDimension,int yBlockDimension,int my_rank)
{

  //printf("xBlock:%i\n",xBlockDimension); fflush(stdin);
  for (int y=0; y<=yBlockDimension+1; y++) {
    for (int x=0; x<=xBlockDimension+1; x++) {
      node[y*(xBlockDimension+2)+x].ix=x*y;
      node[y*(xBlockDimension+2)+x].poten = my_rank;//*10+x;  //initializer
      //  printf("node[%d][%d]: %d\n",x,y,node[x,y].ix); fflush(stdin);
    }
  }
  printf("local_x:%d\t local_y:%d\n",local_x,local_y);
}

double Potential_inside( int maxXCount, node_t node[],double ep0, double deltaX ,double deltaY, int xBlockDimension,int yBlockDimension, int local_x,int n,int m)
{ 
  //double Ne = 1e-9; /* N corresponds to accuracy 
  double error = 0;
  double A[n*m];
  //double k[];
  
  for(int x=2; x<=xBlockDimension-1; x++) {
    for(int y=2; y<=yBlockDimension-1; y++){
      //    // k[i] = (elec[i].density*elec[i].q+Arp[i].density*Arp[i].q)/ep0 *1.e+3;
      
      A[TD2OD(x,y)] = (node[TD2OD(x-1,y)].poten+node[TD2OD(x+1,y)].poten+node[TD2OD(x,y-1)].poten+node[TD2OD(x,y+1)].poten)/4;
     
      error +=  pow((A[TD2OD(x,y)] - node[TD2OD(x,y)].poten)/A[TD2OD(x,y)], 2);
    }
  } 
  for(int x=2; x<=xBlockDimension-1; x++){ 
    for(int y=2; y<=yBlockDimension-1; y++){
      node[TD2OD(x,y)].poten = A[TD2OD(x,y)];
    
      //    // printf("pot:%f\n",A[x,y]);
    }
  }
    
  return error;
}

double Potential_borders( int maxXCount,node_t node[],double ep0, double deltaX ,double deltaY, int xBlockDimension,int yBlockDimension, int local_x, int local_y, int n, int m)
{ 
  //double Ne = 1e-9; /* N corresponds to accuracy 
  double error = 0;
  double A[n*m];

  for(int x=1; x<=xBlockDimension; x++) {
    for(int y=1; y<=yBlockDimension; y++){

  //    if(x>1 && x<xBlockDimension && y>1 && y<yBlockDimension) {A[TD2OD(x,y)]=5; continue;}       // very inefficient, has to be changed ASAP
      
      if(local_y==yBlockDimension && y==yBlockDimension) {                    // UP BORDER 
	A[TD2OD(x,y)]=10;  continue;
      }
      else if(local_x==0 && x==1) {               // LEFT BORDER
	A[TD2OD(x,y)]=0;   continue;
      }
      else if(local_x==xBlockDimension && x==xBlockDimension) { // RIGHT BORDER
	A[TD2OD(x,y)]=10;  continue;
      }
      else if(local_y==0 && y==1) { // DOWN BORDER
	A[TD2OD(x,y)]=0;   continue;
      }
      
      A[TD2OD(x,y)] = (node[TD2OD(x-1,y)].poten+node[TD2OD(x+1,y)].poten+node[TD2OD(x,y-1)].poten+node[TD2OD(x,y+1)].poten)/4;
      error +=  pow((A[TD2OD(x,y)] - node[TD2OD(x,y)].poten)/A[TD2OD(x,y)], 2);
    }
  }

  for(int x=1; x<=xBlockDimension; x++){ 
    for(int y=1; y<=yBlockDimension; y++){
     // if(x>1 && x<xBlockDimension && y>1 && y<yBlockDimension) continue;       // very inefficient, has to be changed ASAP	    
      node[TD2OD(x,y)].poten = A[TD2OD(x,y)];
    }
  }
  
  return error; 
}
  
void output(node_t nd[], int no, int xBlockDimension,int yBlockDimension)
{
  string fname = "output_data_rank";
  fname += to_string(no);
  fname += ".dat";
  ofstream pot(fname);

  // for(int y=0; y<=yBlockDimension+1; y++){
  //   for(int x=0; x<=xBlockDimension+1; x++){
  //     int new_index=y*(xBlockDimension+2)+x;
     
  //     pot << x << "\t" << y
  // 	//	 << "\t" << nd[x][y].ix
  // 	  << "\t" << nd[new_index].poten
  // 	// << "\t" << nd[i].field
  // 	// << "\t" << elec[i].density
  // 	// << "\t" << Arp[i].density
  // 	// << "\t" << elec[i].flux
  // 	// << "\t" << Arp[i].flux
  // 	  << endl;
  //   }
  // }

  for(int y=yBlockDimension; y>=1; y--){
    for(int x=1; x<=xBlockDimension; x++){
    
      int new_index=y*(xBlockDimension+2)+x;
      pot << nd[new_index].poten << "\t";
    }
    pot << endl;
  }
  
  pot.close();
}


