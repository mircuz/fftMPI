	/**************************************************************************************************************
	 * 																											  *
	 * 								2D FFT with Pencil Decomposition in MPI Space								  *
	 * 																											  *
	 **************************************************************************************************************
																						Author: Dr. Mirco Meazzo */
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <cstdio>
#include <time.h>
#include "remap3d_wrap.h"    // To perform 3D remapping
#include "fft_support.h"


#define MODES 4;

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
  nx = modes +1;	ny = modes+4;	nz = 2*modes+1;
  int nxd_AA = (nx*3)/2;
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

  nfast = nzd;
  nmid = nxd;
  nslow = ny;

  // partitioning in z-pencil
  int out_klo = (int) 1.0*ipfast*nfast/npfast;					// K fast
  int out_khi = (int) 1.0*(ipfast+1)*nfast/npfast - 1;
  int out_ilo = (int) 1.0*ipmid*nmid/npmid;						// I med
  int out_ihi = (int) 1.0*(ipmid+1)*nmid/npmid - 1;
  int out_jlo = (int) 1.0*ipslow*nslow/npslow;					// J slow
  int out_jhi = (int) 1.0*(ipslow+1)*nslow/npslow - 1;

  printf("[Z-PENCIL] (k,i,j order)\t"
		  "On rank %d the coordinates are: "
		  "(%d,%d,%d) -> (%d,%d,%d)\n", rank, out_ilo, out_jlo, out_klo, out_ihi, out_jhi, out_khi );

  void *remap_zpencil, *remap_xpencil, *remap_ypencil;
  int nqty, permute, memoryflag, sendsize, recvsize;
  nqty = 2;			// Use couples of real numbers per grid point
  permute = 2;  		// From x-contiguous to z-contiguous arrays
  memoryflag = 1;		// Self-allocate the buffers


  /******************************************* Size Variables ******************************************/
  int insize = (in_ihi-in_ilo+1) * (in_jhi-in_jlo+1) * (in_khi-in_klo+1);
  int outsize = (out_ihi-out_ilo+1) * (out_jhi-out_jlo+1) * (out_khi-out_klo+1);
  int remapsize = (insize > outsize) ? insize : outsize;
  int elem_per_proc = insize*2;

  // Alloc the arrays
  int* displs = (int *)malloc(size*sizeof(int));
  int* scounts = (int *)malloc(size*sizeof(int));
  int* receive = (int *)malloc(size*sizeof(int));

  // Setup matrix
  int modes_per_proc[size];
  modes_per_proc[rank] = (in_jhi-in_jlo+1) * (in_khi-in_klo+1);
  MPI_Allgather(&modes_per_proc[rank],1,MPI_INT,modes_per_proc,1,MPI_INT, MPI_COMM_WORLD);
  // Scattering parameters
  int offset=0;
  for (int i=0; i<size; ++i) {
  	  scounts[i] = modes_per_proc[i]*nxd*2;
  	  receive[i] = scounts[i];
  	  displs[i] = offset ;
  	  offset += scounts[i];
  }

  //printf("modes_ proc %d on rank %d\n", modes_per_proc[rank], rank);
  //printf("scoutn %d & disps %d on rank %d\n", scounts[rank], displs[rank], rank);


  /******************************************** Memory Alloc *******************************************/
  FFT_SCALAR *u = (FFT_SCALAR *) malloc(remapsize*sizeof(FFT_SCALAR)*2);
  FFT_SCALAR *v = (FFT_SCALAR *) malloc(remapsize*sizeof(FFT_SCALAR)*2);
  FFT_SCALAR *w = (FFT_SCALAR *) malloc(remapsize*sizeof(FFT_SCALAR)*2);
  FFT_SCALAR *uu = (FFT_SCALAR *) malloc(remapsize*sizeof(FFT_SCALAR)*2);
  FFT_SCALAR *uv = (FFT_SCALAR *) malloc(remapsize*sizeof(FFT_SCALAR)*2);
  FFT_SCALAR *vv = (FFT_SCALAR *) malloc(remapsize*sizeof(FFT_SCALAR)*2);
  FFT_SCALAR *vw = (FFT_SCALAR *) malloc(remapsize*sizeof(FFT_SCALAR)*2);
  FFT_SCALAR *ww = (FFT_SCALAR *) malloc(remapsize*sizeof(FFT_SCALAR)*2);
  FFT_SCALAR *uw = (FFT_SCALAR *) malloc(remapsize*sizeof(FFT_SCALAR)*2);
  FFT_SCALAR *sendbuf = (FFT_SCALAR *) malloc(sendsize*sizeof(FFT_SCALAR)*2);
  FFT_SCALAR *recvbuf = (FFT_SCALAR *) malloc(recvsize*sizeof(FFT_SCALAR)*2);

  FFT_SCALAR *u_ref = (FFT_SCALAR *) malloc(remapsize*sizeof(FFT_SCALAR)*2);
  // Declare variables, on all procs, needed to Scatter data
  FFT_SCALAR *V, *U, *W;

  if (rank == 0) {
	  // Allocate mememory needed to Scatter data, only on the broadcaster
	  U = (FFT_SCALAR*) malloc( nxd*ny*nzd*2* sizeof(FFT_SCALAR));
	  V = (FFT_SCALAR*) malloc( nxd*ny*nzd*2* sizeof(FFT_SCALAR));
	  W = (FFT_SCALAR*) malloc( nxd*ny*nzd*2* sizeof(FFT_SCALAR));
	  read_data_and_apply_AA(nx, ny, nz, nxd, nzd, U, V, W);

	  printf("Reading and Antialiasing completed...\n");
  }

  //Send chunks of array Velocity to all processors
  MPI_Scatterv(U, scounts, displs, MPI_DOUBLE, u, receive[rank] , MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Scatterv(V, scounts, displs, MPI_DOUBLE, v, receive[rank] , MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Scatterv(W, scounts, displs, MPI_DOUBLE, w, receive[rank] , MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  //print_array( u, insize, i_length, rank, "Initialized Values");
  //print_y_pencil(nx, ny, nz, u, rank, displs[rank], scounts[rank], 0);


  /************************************************ backward FFTs *********************************************/

  if (rank == 0) printf("Starting Backward transformations...\n");
  // ------------------------------------------- Setup z-Transpose --------------------------------------------
  remap3d_create( remap_comm , &remap_zpencil);
  remap3d_setup( remap_zpencil,
      		  	  in_ilo,  in_ihi,  in_jlo, in_jhi,  in_klo,  in_khi,
				  out_ilo,  out_ihi,  out_jlo, out_jhi,  out_klo,  out_khi,
      			  nqty, permute, memoryflag, &sendsize, &recvsize);
  // -----------------------------------------------------------------------------------------------------------
  // Backward FFT#1
  MPI_Barrier( MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  double timer_b1 = 0.0;
  timer_b1 -= MPI_Wtime();
  b_FFT( u, elem_per_proc, i_length );	MPI_Barrier( MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  b_FFT( v, elem_per_proc, i_length );	MPI_Barrier( MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  b_FFT( w, elem_per_proc, i_length );	MPI_Barrier( MPI_COMM_WORLD); // @suppress("Symbol is not resolved")

  timer_b1 += MPI_Wtime();

  // Transpose in z-pencil
  double timer_trasp_z = 0.0, TIMER_TRASP_z = 0.0;
  timer_trasp_z -= MPI_Wtime();
  remap3d_remap(remap_zpencil,u,u,sendbuf,recvbuf);	MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  remap3d_remap(remap_zpencil,v,v,sendbuf,recvbuf);	MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  remap3d_remap(remap_zpencil,w,w,sendbuf,recvbuf);	MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  timer_trasp_z += MPI_Wtime();


  // Backward FFT#2
  double timer_b2 = 0.0;
  MPI_Barrier( MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  timer_b2 -= MPI_Wtime();
  b_FFT( u, elem_per_proc, k_length );	MPI_Barrier( MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  b_FFT( v, elem_per_proc, k_length );	MPI_Barrier( MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  b_FFT( w, elem_per_proc, k_length );	MPI_Barrier( MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  timer_b2 += MPI_Wtime();


  //Finalize plan
  remap3d_destroy(remap_zpencil);
  //print_y_pencil(nx, ny, nz, u, rank, displs[rank], scounts[rank], 0);
  //print_array( u, insize, k_length, rank, "First Transpose of U performed");


  /************************************************ Convolutions *********************************************/
  // Operations
  double timer_conv = 0.0;
  timer_conv -= MPI_Wtime();
  for ( int i = 0; i < elem_per_proc; i++) {
	  uu[i] = u[i]*u[i];
	  uv[i] = u[i]*v[i];
	  vv[i] = v[i]*v[i];
	  vw[i] = v[i]*w[i];
	  ww[i] = w[i]*w[i];
	  uw[i] = u[i]*w[i];
  }
  timer_conv += MPI_Wtime();
  //print_array( uu, insize, k_length, rank, "UU performed");
  MPI_Barrier( MPI_COMM_WORLD); // @suppress("Symbol is not resolved")

  //print_y_pencil(nx, ny, nz, uu, rank, displs[rank], scounts[rank], 0);
  if (rank == 0) printf("Completed\nStarting Forward transformations...\n");

  /************************************************ forward FFTs *********************************************/
  // ---------------------------------------- Setup x-Transpose -----------------------------------------
  remap3d_create( remap_comm , &remap_xpencil);
  permute = 1; 		// From z-contiguous to x-contiguous arrays
  remap3d_setup( remap_xpencil,
		  	  	  out_klo,  out_khi, out_ilo,  out_ihi,  out_jlo, out_jhi,
				  in_klo,  in_khi, in_ilo,  in_ihi,  in_jlo, in_jhi,
				  nqty, permute, memoryflag, &sendsize, &recvsize);
  // -----------------------------------------------------------------------------------------------------------

  // Forward FFT#1
  double timer_f1 = 0.0;
  MPI_Barrier( MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  timer_f1 -= MPI_Wtime();
  f_FFT( u, elem_per_proc, k_length );

  f_FFT( uu, elem_per_proc, k_length );	 MPI_Barrier( MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  f_FFT( uv, elem_per_proc, k_length );	 MPI_Barrier( MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  f_FFT( vv, elem_per_proc, k_length );	 MPI_Barrier( MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  f_FFT( vw, elem_per_proc, k_length );	 MPI_Barrier( MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  f_FFT( ww, elem_per_proc, k_length );	 MPI_Barrier( MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  f_FFT( uw, elem_per_proc, k_length );	 MPI_Barrier( MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  timer_f1 += MPI_Wtime();


  // Transpose to x-pencil
  double timer_trasp_x = 0.0, TIMER_TRASP_x = 0.0;
  timer_trasp_x -= MPI_Wtime();
  remap3d_remap(remap_xpencil,u,u,sendbuf,recvbuf); 	MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")

  remap3d_remap(remap_xpencil,uu,uu,sendbuf,recvbuf); 	MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  remap3d_remap(remap_xpencil,uv,uv,sendbuf,recvbuf);	MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  remap3d_remap(remap_xpencil,vv,vv,sendbuf,recvbuf);	MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  remap3d_remap(remap_xpencil,vw,vw,sendbuf,recvbuf);	MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  remap3d_remap(remap_xpencil,ww,ww,sendbuf,recvbuf);	MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  remap3d_remap(remap_xpencil,uw,uw,sendbuf,recvbuf);	MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  timer_trasp_x += MPI_Wtime();


  // Forward FFT#2
  double timer_f2 = 0.0;
  MPI_Barrier( MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  timer_f2 -= MPI_Wtime();
  f_FFT( u, elem_per_proc, i_length );

  f_FFT( uu, elem_per_proc, i_length );	 MPI_Barrier( MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  f_FFT( uv, elem_per_proc, i_length );	 MPI_Barrier( MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  f_FFT( vv, elem_per_proc, i_length );	 MPI_Barrier( MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  f_FFT( vw, elem_per_proc, i_length );	 MPI_Barrier( MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  f_FFT( ww, elem_per_proc, i_length );	 MPI_Barrier( MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  f_FFT( uw, elem_per_proc, i_length );	 MPI_Barrier( MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  timer_f2 += MPI_Wtime();


  // Finalize plan
  remap3d_destroy(remap_xpencil);
  MPI_Barrier( MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  //print_array( uu, insize, i_length, rank, "Results U");
  if (rank == 0) printf("Completed\nStarting dealiasing operations\n");


  /*********************************************** Modes cutting ********************************************/
  double TIMER_AA = 0.0;
  TIMER_AA -= MPI_Wtime();

  // Alloc memory for the global output
  FFT_SCALAR *UU, *UV, *VV, *VW, *WW, *UW;
  UU = (FFT_SCALAR*) malloc( nfast*nmid*nslow*2* sizeof(FFT_SCALAR));
  UV = (FFT_SCALAR*) malloc( nfast*nmid*nslow*2* sizeof(FFT_SCALAR));
  VV = (FFT_SCALAR*) malloc( nfast*nmid*nslow*2* sizeof(FFT_SCALAR));
  VW = (FFT_SCALAR*) malloc( nfast*nmid*nslow*2* sizeof(FFT_SCALAR));
  WW = (FFT_SCALAR*) malloc( nfast*nmid*nslow*2* sizeof(FFT_SCALAR));
  UW = (FFT_SCALAR*) malloc( nfast*nmid*nslow*2* sizeof(FFT_SCALAR));

  // Gather all data on rank 0
  MPI_Gatherv( u, receive[rank], MPI_DOUBLE, U, scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")

  //printf("RANK %d\t sendcout: %d\t receivecount: %d\t displs %d\n", rank, receive[rank], scounts[rank], displs[rank]);
  MPI_Gatherv( uu, receive[rank], MPI_DOUBLE, UU, scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  MPI_Gatherv( uv, receive[rank], MPI_DOUBLE, UV, scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  MPI_Gatherv( vv, receive[rank], MPI_DOUBLE, VV, scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  MPI_Gatherv( vw, receive[rank], MPI_DOUBLE, VW, scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  MPI_Gatherv( ww, receive[rank], MPI_DOUBLE, WW, scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  MPI_Gatherv( uw, receive[rank], MPI_DOUBLE, UW, scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")


  /**************************************** Dealias and Transpose dataset ****************************************/
  if (rank == 0) {
	  dealiasing( nx, ny, nz, nxd, nzd, U);

	  dealiasing( nx, ny, nz, nxd, nzd, UU);
	  dealiasing( nx, ny, nz, nxd, nzd, UV);
	  dealiasing( nx, ny, nz, nxd, nzd, VV);
	  dealiasing( nx, ny, nz, nxd, nzd, VW);
	  dealiasing( nx, ny, nz, nxd, nzd, WW);
	  dealiasing( nx, ny, nz, nxd, nzd, UW);


	  transpose_on_rank0( nx, ny, nz, U);

	  transpose_on_rank0( nx,  ny, nz, UU);
	  transpose_on_rank0( nx,  ny, nz, UV);
	  transpose_on_rank0( nx,  ny, nz, VV);
	  transpose_on_rank0( nx,  ny, nz, VW);
	  transpose_on_rank0( nx,  ny, nz, WW);
	  transpose_on_rank0( nx,  ny, nz, UW);
  }


  /********************************** Setup asymetric factors for scattering **********************************/
  // Setup matrix
   modes_per_proc[size];
  for (int i = 0; i < size; i++){
	  modes_per_proc[i] = 0;
	  displs[i] = 0;
  }
  // Set modes per processor
  cores_handler( nx*nz, size, modes_per_proc);
  // Scattering parameters
  offset=0;
  for (int i=0; i<size; ++i) {
	  scounts[i] = modes_per_proc[i]*ny*2;
	  receive[i] = scounts[i];
	  displs[i] = offset ;
	  offset += modes_per_proc[i] *ny*2;
  }


  /************************************************ Data scattering ***********************************************/
  MPI_Scatterv(U, scounts, displs, MPI_DOUBLE, u, receive[rank] , MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Scatterv(UU, scounts, displs, MPI_DOUBLE, uu, receive[rank] , MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Scatterv(UV, scounts, displs, MPI_DOUBLE, uv, receive[rank] , MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Scatterv(VV, scounts, displs, MPI_DOUBLE, vv, receive[rank] , MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Scatterv(VW, scounts, displs, MPI_DOUBLE, vw, receive[rank] , MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Scatterv(WW, scounts, displs, MPI_DOUBLE, ww, receive[rank] , MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Scatterv(UW, scounts, displs, MPI_DOUBLE, uw, receive[rank] , MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  TIMER_AA += MPI_Wtime();

  //print_y_pencil(nx, ny, nz, u, rank, displs[rank], scounts[rank], 0);

  /************************************************ y-Transpose *********************************************/
/*  // IN
  nfast = nx;
  nmid = ny;
  nslow = nz;

  // partitioning in x-pencil
  in_ilo = (int) 1.0*ipfast*nfast/npfast;					// K fast
  in_ihi = (int) 1.0*(ipfast+1)*nfast/npfast - 1;
  in_jlo = (int) 1.0*ipmid*nmid/npmid;						// I med
  in_jhi = (int) 1.0*(ipmid+1)*nmid/npmid - 1;
  in_klo = (int) 1.0*ipslow*nslow/npslow;					// J slow
  in_khi = (int) 1.0*(ipslow+1)*nslow/npslow - 1;

  // OUT
  nfast = ny;
    nmid = nz;
    nslow = nx;

    // partitioning in y-pencil
    int outy_jlo = (int) 1.0*ipfast*nfast/npfast;						// J fast
    int outy_jhi = (int) 1.0*(ipfast+1)*nfast/npfast - 1;
    int outy_klo = (int) 1.0*ipmid*nmid/npmid;						// I med
    int outy_khi = (int) 1.0*(ipmid+1)*nmid/npmid - 1;
    int outy_ilo = (int) 1.0*ipslow*nslow/npslow;						// K slow
    int outy_ihi = (int) 1.0*(ipslow+1)*nslow/npslow - 1;

  printf("[Y-PENCIL] (j,i,k order)\t"

    		  "On rank %d the coordinates are: "
  		  "(%d,%d,%d) -> (%d,%d,%d)\n", rank, outy_ilo, outy_jlo, outy_klo, outy_ihi, outy_jhi, outy_khi );

  // -------------------------------------------- Setup y-Transpose ------------------------------------------
  remap3d_create( remap_comm , &remap_ypencil);
  permute = 1;
  remap3d_setup( remap_ypencil,
		  	  	  in_ilo,  in_ihi,  in_jlo, in_jhi,  in_klo,  in_khi,
				  outy_ilo,  outy_ihi,  outy_jlo, outy_jhi,  outy_klo,  outy_khi,
				  nqty, permute, memoryflag, &sendsize, &recvsize);
  //----------------------------------------------------------------------------------------------------------
  double timer_trasp_y = 0.0, TIMER_TRASP_y = 0.0;
  timer_trasp_y -= MPI_Wtime();
  remap3d_remap(remap_ypencil,u,u,sendbuf,recvbuf); 	MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")

 	int stride_nx=0; int stride_nz = 0;
    if (rank == 0) {
    for (int i = 0; i < elem_per_proc; i++) {
  	  if ( i % (ny*2) == 0) {
  		  printf("========(nx= %d, nz= %d)=======\n", stride_nx, stride_nz);
  		  stride_nx ++;
  		  if ( stride_nx % nx == 0) {
  			  stride_nx =0;
  			  stride_nz ++;
  		  }
  	  }
  	 printf("u[%d]= %g\n", (i+rank*nx*ny*nz), u[i]);
    }
    }

  remap3d_remap(remap_ypencil,uu,uu,sendbuf,recvbuf); 	MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  remap3d_remap(remap_ypencil,uv,uv,sendbuf,recvbuf);	MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  remap3d_remap(remap_ypencil,vv,vv,sendbuf,recvbuf);	MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  remap3d_remap(remap_ypencil,vw,vw,sendbuf,recvbuf);	MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  remap3d_remap(remap_ypencil,ww,ww,sendbuf,recvbuf);	MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  remap3d_remap(remap_ypencil,uw,uw,sendbuf,recvbuf);	MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  timer_trasp_y += MPI_Wtime();


  //Finalize plan
  remap3d_destroy(remap_ypencil);
  insize = (outy_ihi-outy_ilo+1) * (outy_jhi-outy_jlo+1) * (outy_khi-outy_klo+1);
  //print_array( u, insize, ny, rank, "y-Transpose of U performed");

*/
  /************************************************ Print Stats *********************************************/
  // Gather all stats
  double TIMER_b1, TIMER_b2, TIMER_f1, TIMER_f2, TIMER_conv;
  MPI_Allreduce(&timer_trasp_x, &TIMER_TRASP_x,1,MPI_DOUBLE,MPI_MAX,remap_comm); // @suppress("Symbol is not resolved")
  MPI_Allreduce(&timer_trasp_z, &TIMER_TRASP_z,1,MPI_DOUBLE,MPI_MAX,remap_comm); // @suppress("Symbol is not resolved")
 // MPI_Allreduce(&timer_trasp_y, &TIMER_TRASP_y,1,MPI_DOUBLE,MPI_MAX,remap_comm); // @suppress("Symbol is not resolved")
  MPI_Allreduce(&timer_b1, &TIMER_b1,1,MPI_DOUBLE,MPI_MAX,remap_comm); // @suppress("Symbol is not resolved")
  MPI_Allreduce(&timer_b2, &TIMER_b2,1,MPI_DOUBLE,MPI_MAX,remap_comm); // @suppress("Symbol is not resolved")
  MPI_Allreduce(&timer_f1, &TIMER_f1,1,MPI_DOUBLE,MPI_MAX,remap_comm); // @suppress("Symbol is not resolved")
  MPI_Allreduce(&timer_f2, &TIMER_f2,1,MPI_DOUBLE,MPI_MAX,remap_comm); // @suppress("Symbol is not resolved")
  MPI_Allreduce(&timer_conv, &TIMER_conv,1,MPI_DOUBLE,MPI_MAX,remap_comm); // @suppress("Symbol is not resolved")
  MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")

  // Print stats
  if (rank == 0) {
	  printf("\n-----------------------------------------------------------\n");
	  printf("%lgs employed to perform 2D FFT (backward) \n", TIMER_b1 +TIMER_b2);
	  printf("%lgs employed to transpose the array (z-pencil) \n", TIMER_TRASP_z);
	  printf("%lgs employed to perform 2D FFT (forward) \n", TIMER_f1 +TIMER_f2);
  	  printf("%lgs employed to transpose the array (x-pencil) \n", TIMER_TRASP_x);
  	  printf("%lgs employed to perform convolutions \n", TIMER_conv);
  	  printf("%lgs employed to gather, cut modes, transpose & scatter data \n", TIMER_AA);
  	  printf("-----------------------------------------------------------\n\n");
/*  	  // Disk files
  	  printf("Saving data on disk...\n");
  	  FILE *UU_dat, *UV_dat, *VV_dat, *VW_dat, *WW_dat, *UW_dat;
  	  UU_dat = fopen( "uu.dat", "w+");		UV_dat = fopen( "uv.dat", "w+");
  	  VV_dat = fopen( "vv.dat", "w+");		VW_dat = fopen( "vw.dat", "w+");
  	  WW_dat = fopen( "ww.dat", "w+");		UW_dat = fopen( "uw.dat", "w+");
  	  for (int i = 0; i < nfast*nmid*nslow*2; i++ ) {
  		  fprintf( UU_dat, "%lf\n", UU[i]);		fprintf( UV_dat, "%lf\n", UV[i]);
  		  fprintf( VV_dat, "%lf\n", VV[i]);		fprintf( VW_dat, "%lf\n", VW[i]);
  		  fprintf( WW_dat, "%lf\n", WW[i]);		fprintf( UW_dat, "%lf\n", UW[i]);
  	  }
*/  	  printf("Process Ended\n");
  }

  /**************************************** Release Mem & Finalize MPI *************************************/
  free(u);	free(v);	free(w);
  free(uu);	free(uv);	free(vv);	free(vw);	free(ww);	free(uw);
  free(U);	free(V);	free(W);
  free(UU); free(UV);	free(VV);	free(VW);	free(WW);	free(UW);
  free(recvbuf);
  free(sendbuf);
  MPI_Finalize();
}
