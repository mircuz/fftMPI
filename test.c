#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>


void printarr(int *data, int nx, int ny, int nz, int rank, int, char *str);
int *allocarray(int nx, int ny, int nz);

#define MODES 4
int main(int narg, char **args) {

	// setup MPI
	  MPI_Init(&narg,&args);
	  MPI_Comm remap_comm; // @suppress("Type cannot be resolved")
	  MPI_Comm_dup( MPI_COMM_WORLD, &remap_comm ); // @suppress("Symbol is not resolved")
	  MPI_Comm world = MPI_COMM_WORLD; // @suppress("Symbol is not resolved") // @suppress("Type cannot be resolved")
	  int rank,size;
	  MPI_Comm_size(world,&size);
	  MPI_Comm_rank(world,&rank);

	  // Antialias along x
	  int modes, nfast,nmid,nslow, nx,ny,nz;
	  modes = MODES;
	  nx = modes/2;	ny = modes;	nz = modes+1;
	  int nxd_AA = (nx*3)/2 +1;
	  // fftFitting
	  int nxd = 4; int nzd = 4;
	  while ( nxd < nxd_AA ) {
		  nxd = nxd*2;
	  }
	  while ( nzd < nz ) {
		  nzd = nzd*2;
	  }

	  // Length of the array along directions
	  int i_length = nxd;
	  int j_length = ny;
	  int k_length = nzd;

	  // TOTAL Modes
	  nfast = nxd;
	  nmid = ny;
	  nslow = nzd;

	  // Algorithm to factor Nprocs into roughly cube roots
	  int npfast,npmid,npslow;
	  npfast= 1;
	  int npmidslow = size / npfast;
	  npmid = (int) sqrt(npmidslow);
	  while (npmid < npmidslow) {
	    if (npmidslow % npmid == 0) break;
	    npmid++;
	  }
	  npslow = size / npfast / npmid;


	  if (rank == 0) {
	  	  printf("\n========================================================================================\n"
	  			  "\t2D FFT with %dx%dx%d total modes (%d,%d,%d) on %d procs, %dx%dx%d grid\t\n"
	  			"========================================================================================\n\n",
	  			  nfast,nmid,nslow,nx,ny,nz,size,npfast,npmid,npslow);
	    }

	  if ((ny/npmid) < 1) {
		  perror("\n\n\nInvalid Y Grid decomposition\nAborting simulation...\n\n\n");
		  abort();
	  }
	  if ((nz/npslow) < 1) {
	 	  perror("\n\n\nInvalid Z Grid decomposition\nAborting simulation...\n\n\n");
	 	  abort();
	   }

	  /******************************************** Remap Variables *******************************************/
	  // partitioning in x-pencil
	  int ipfast = rank % npfast;
	  int ipmid = (rank/npfast) % npmid;
	  int ipslow = rank / (npfast*npmid);
	  int in_ilo, in_ihi, in_jlo, in_jhi, in_klo, in_khi;

	  in_ilo = (int) 1.0*ipfast*nfast/npfast;						// I fast
	  in_ihi = (int) 1.0*(ipfast+1)*nfast/npfast - 1;
	  in_jlo = (int) 1.0*ipmid*nmid/npmid;							// J med
	  in_jhi = (int) 1.0*(ipmid+1)*nmid/npmid - 1;
	  in_klo = (int) 1.0*ipslow*nslow/npslow;						// K slow
	  in_khi = (int) 1.0*(ipslow+1)*nslow/npslow - 1;

	  printf("[X-PENCIL] (i,j,k order)\t"
			  "On rank %d the coordinates are: "
			  "(%d,%d,%d) -> (%d,%d,%d)\n", rank, in_ilo, in_jlo, in_klo, in_ihi, in_jhi, in_khi );

	  int *contiguous_y = (int *) malloc(sizeof(int)*size);
	  int *contiguous_z = (int *) malloc(sizeof(int)*size);
	  contiguous_y[rank] = (in_jhi-in_jlo+1);
	  contiguous_z[rank] = (in_khi-in_klo+1);
	  MPI_Allgather(&contiguous_y[rank],1,MPI_INT,contiguous_y,1,MPI_INT, MPI_COMM_WORLD);
	  MPI_Allgather(&contiguous_z[rank],1,MPI_INT,contiguous_z,1,MPI_INT, MPI_COMM_WORLD);

	  if (rank == 0){
	  for(int i = 0; i < size; i++) {
		  printf("cont_y %d, rank %d\n", contiguous_z[i], i);
	  }
	  }

	  if (rank == 0) {
		  int *arr = allocarray(4*2,4,8);
		  for(int k=0; k<8; k++) {
			  for (int j=0; j<4;j++) {
				  for (int i=0; i<8; i++) {
					  arr[i+j*8+k*8*4] = i+j*8+k*8*4;
				  }
			  }
		  }
		  printarr(arr, 8, 4, 8, rank, 0, "Starting array");

		  MPI_Datatype vector[size], contiguous[size];
		  int bytes_stride = sizeof(int)*2*nxd*ny;

		  for (int i = 0; i < size; i++) {
			  MPI_Type_contiguous(2*nxd*contiguous_y[i], MPI_INT, &contiguous[i]);
			  MPI_Type_create_hvector(contiguous_z[i], 1, bytes_stride, contiguous[i], &vector[i]);
			  MPI_Type_commit(&vector[i]);
		  }

		  MPI_Send(&arr[176], 1, vector[8], 8, 10, MPI_COMM_WORLD);

		  MPI_Type_free(vector);
	  }

    if (rank == 8) {
    	int *arr_recv = (int*)malloc(2*nxd*(in_jhi-in_jlo+1)*(in_khi-in_klo+1)*sizeof(int));
    	MPI_Recv(arr_recv, 2*nxd*(in_jhi-in_jlo+1)*(in_khi-in_klo+1), MPI_INT, 0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    	for(int i=0; i < 2*nxd*(in_jhi-in_jlo+1)*(in_khi-in_klo+1); i ++){
    		printf("%d: %d\t\t",i, arr_recv[i]);
    	}
    }

    MPI_Finalize();
    return 0;
}

void printarr(int *data, int nx, int ny, int nz, int rank, int desidered_rank, char *str) {
    if(rank == desidered_rank){
	printf("-- %s --\n", str);
    for(int k=0; k<nz; k++) {
    	printf("\n\n-----%d------\n",k);
    	for (int j=0; j<ny;j++) {
    		for (int i=0; i<nx; i++) {
    			printf("%3d\t", data[i+j*nx+k*nx*ny]);
    		}
        printf("\n");
    	}
    }
    }
}

int *allocarray(int nx, int ny, int nz) {
	int* arr = (int*)malloc(sizeof(int)*nz*nx*ny);
    return arr;
}
