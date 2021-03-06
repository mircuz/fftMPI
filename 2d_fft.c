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
#include <fftw3.h>


#define MODES 8;

int main(int narg, char **args) {

  // setup MPI
  MPI_Init(&narg,&args);
  MPI_Comm remap_comm; // @suppress("Type cannot be resolved")
  MPI_Comm_dup( MPI_COMM_WORLD, &remap_comm ); // @suppress("Symbol is not resolved")
  int rank,size;
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  // Antialias along x
  int modes, nfast,nmid,nslow, nx,ny,nz;
  modes = MODES;
  nx = modes/2 +1;	ny = modes;	nz = modes+1;
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
  int x_ilo, x_ihi, x_jlo, x_jhi, x_klo, x_khi;

  x_ilo = (int) 1.0*ipfast*nfast/npfast;						// I fast
  x_ihi = (int) 1.0*(ipfast+1)*nfast/npfast - 1;
  x_jlo = (int) 1.0*ipmid*nmid/npmid;							// J med
  x_jhi = (int) 1.0*(ipmid+1)*nmid/npmid - 1;
  x_klo = (int) 1.0*ipslow*nslow/npslow;						// K slow
  x_khi = (int) 1.0*(ipslow+1)*nslow/npslow - 1;

  printf("[X-PENCIL] (i,j,k order)\t"
		  "On rank %d the coordinates are: "
		  "(%d,%d,%d) -> (%d,%d,%d)\n", rank, x_ilo, x_jlo, x_klo, x_ihi, x_jhi, x_khi );

  nfast = nzd;
  nmid = nxd;
  nslow = ny;

  // partitioning in z-pencil
  int z_klo = (int) 1.0*ipfast*nfast/npfast;					// K fast
  int z_khi = (int) 1.0*(ipfast+1)*nfast/npfast - 1;
  int z_ilo = (int) 1.0*ipmid*nmid/npmid;						// I med
  int z_ihi = (int) 1.0*(ipmid+1)*nmid/npmid - 1;
  int z_jlo = (int) 1.0*ipslow*nslow/npslow;					// J slow
  int z_jhi = (int) 1.0*(ipslow+1)*nslow/npslow - 1;

  printf("[Z-PENCIL] (k,i,j order)\t"
		  "On rank %d the coordinates are: "
		  "(%d,%d,%d) -> (%d,%d,%d)\n", rank, z_ilo, z_jlo, z_klo, z_ihi, z_jhi, z_khi );

  void *remap_zpencil, *remap_xpencil, *remap_ypencil;
  int nqty, permute, memoryflag, sendsize, recvsize;
  nqty = 2;			// Use couples of real numbers per grid point
  permute = 2;  		// From x-contiguous to z-contiguous arrays
  memoryflag = 1;		// Self-allocate the buffers


  /******************************************* Size Variables ******************************************/
  int insize = (x_ihi-x_ilo+1) * (x_jhi-x_jlo+1) * (x_khi-x_klo+1);
  int outsize = (z_ihi-z_ilo+1) * (z_jhi-z_jlo+1) * (z_khi-z_klo+1);
  int remapsize = (insize > outsize) ? insize : outsize;
  int elem_per_proc = insize*2;

  // Alloc the arrays
  int* displs = (int *)malloc(size*sizeof(int));
  int* scounts = (int *)malloc(size*sizeof(int));
  int* receive = (int *)malloc(size*sizeof(int));
  int* displs_gather = (int *)malloc(size*sizeof(int));
  int* scounts_gather = (int *)malloc(size*sizeof(int));
  int* receive_gather = (int *)malloc(size*sizeof(int));

  // Setup matrix
  int *modes_per_proc = (int *) malloc(sizeof(int)*size);
  modes_per_proc[rank] = (x_jhi-x_jlo+1) * (x_khi-x_klo+1);
  MPI_Allgather(&modes_per_proc[rank],1,MPI_INT,modes_per_proc,1,MPI_INT, MPI_COMM_WORLD);
  // Scattering & Gathering parameters
  int offset=0;
  for (int i=0; i<size; ++i) {
  	  scounts[i] = modes_per_proc[i]*nxd*2;
  	  receive[i] = scounts[i];
  	  displs[i] = offset;
  	  offset += scounts[i];
  }
  //printf("modes_ proc %d on rank %d\n", modes_per_proc[rank], rank);
  //printf("scoutn %d & disps %d on rank %d\n", scounts[rank], displs[rank], rank);


  /********************************** Setup asymetric factors for scattering **********************************/
    // Setup matrix
    int modes_per_proc_scat[size], displs_scat[size], scounts_scat[size], receive_scat[size];
    for (int i = 0; i < size; i++){
  	  modes_per_proc_scat[i] = 0;
  	  displs_scat[i] = 0;
    }
    // Set modes per processor
    cores_handler( nx*nz, size, modes_per_proc_scat);
    // Scattering parameters
    offset=0;
    for (int i=0; i<size; ++i) {
  	  scounts_scat[i] = modes_per_proc_scat[i]*ny*2;
  	  receive_scat[i] = scounts_scat[i];
    	  displs_scat[i] = offset ;
    	  offset += modes_per_proc_scat[i] *ny*2;
    }
    //printf("modes_ proc %d on rank %d\n", modes_per_proc[rank], rank);
    //printf("scoutn %d & disps %d on rank %d\n", scounts[rank], displs[rank], rank);


    /******************************************** Memory Alloc *******************************************/
    FFT_SCALAR *u = (FFT_SCALAR *) malloc(remapsize*sizeof(FFT_SCALAR)*2);
    FFT_SCALAR *v = (FFT_SCALAR *) malloc(remapsize*sizeof(FFT_SCALAR)*2);
    FFT_SCALAR *w = (FFT_SCALAR *) malloc(remapsize*sizeof(FFT_SCALAR)*2);
    FFT_SCALAR *u_read = (FFT_SCALAR *) malloc(remapsize*sizeof(FFT_SCALAR)*2);
    FFT_SCALAR *v_read = (FFT_SCALAR *) malloc(remapsize*sizeof(FFT_SCALAR)*2);
    FFT_SCALAR *w_read = (FFT_SCALAR *) malloc(remapsize*sizeof(FFT_SCALAR)*2);
    FFT_SCALAR *uu = (FFT_SCALAR *) malloc(remapsize*sizeof(FFT_SCALAR)*2);
    FFT_SCALAR *uv = (FFT_SCALAR *) malloc(remapsize*sizeof(FFT_SCALAR)*2);
    FFT_SCALAR *vv = (FFT_SCALAR *) malloc(remapsize*sizeof(FFT_SCALAR)*2);
    FFT_SCALAR *vw = (FFT_SCALAR *) malloc(remapsize*sizeof(FFT_SCALAR)*2);
    FFT_SCALAR *ww = (FFT_SCALAR *) malloc(remapsize*sizeof(FFT_SCALAR)*2);
    FFT_SCALAR *uw = (FFT_SCALAR *) malloc(remapsize*sizeof(FFT_SCALAR)*2);
    FFT_SCALAR *sendbuf = (FFT_SCALAR *) malloc(sendsize*sizeof(FFT_SCALAR)*2);
    FFT_SCALAR *recvbuf = (FFT_SCALAR *) malloc(recvsize*sizeof(FFT_SCALAR)*2);

    if ((u||u_read||v||v_read||w||w_read||uu||uv||vv||vw||ww||uw||sendbuf||recvbuf) == NULL) {
  	  perror(".:Error while allocating memory for remapping variables:.\n");
  	  abort();
    }

    // Declare variables, on all procs, needed to Scatter data
    FFT_SCALAR *V, *U, *W, *U_read, *V_read, *W_read;

    // Allocate mememory needed to Scatter data, only on the broadcaster
    // U
    U_read = (FFT_SCALAR*) malloc( nx*ny*nz*2* sizeof(FFT_SCALAR));
    U = (FFT_SCALAR*) malloc( nxd*ny*nzd*2* sizeof(FFT_SCALAR));
    if( (U||U_read) == NULL) {
  	  perror(".:Error while allocatin broadcaster memory U:.\n");
  	  abort();
    }
    if (rank == 0) {
    	read_data(nx, ny, nz, U_read, "u.dat");
    	apply_AA(nx, ny, nz, nxd, nzd, U, U_read);
    }
    Alltoall( rank, size, x_jlo, x_jhi, x_klo, x_khi, nxd, ny, nzd, U, u, 1);
    MPI_Barrier(MPI_COMM_WORLD);
    free(U);
    if (rank == 0) transpose_on_rank0( nx, ny, nz, U_read);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Scatterv(U_read, scounts_scat, displs_scat, MPI_DOUBLE, u_read, receive_scat[rank] , MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) free(U_read);
    MPI_Barrier(MPI_COMM_WORLD);

    //V
    V_read = (FFT_SCALAR*) malloc( nx*ny*nz*2* sizeof(FFT_SCALAR));
    V = (FFT_SCALAR*) malloc( nxd*ny*nzd*2* sizeof(FFT_SCALAR));
    if( (V||V_read) == NULL) {
    	perror(".:Error while allocating broadcaster memory V:.\n");
    	abort();
    }
    if (rank == 0){
    	read_data(nx, ny, nz, V_read, "v.dat");
    	apply_AA(nx, ny, nz, nxd, nzd, V, V_read);
    }
    Alltoall( rank, size, x_jlo, x_jhi, x_klo, x_khi, nxd, ny, nzd, V, v, 1);
    MPI_Barrier(MPI_COMM_WORLD);
    free(V);
    if (rank == 0) transpose_on_rank0( nx, ny, nz, V_read);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Scatterv(V_read, scounts_scat, displs_scat, MPI_DOUBLE, v_read, receive_scat[rank] , MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) free(V_read);
    MPI_Barrier(MPI_COMM_WORLD);

    //W
    W_read = (FFT_SCALAR*) malloc( nx*ny*nz*2* sizeof(FFT_SCALAR));
    W = (FFT_SCALAR*) malloc( nxd*ny*nzd*2* sizeof(FFT_SCALAR));
    if( (W||W_read) == NULL) {
    	perror(".:Error while allocating broadcaster memory W:.\n");
    	abort();
    }
    if (rank == 0){
    	read_data(nx, ny, nz, W_read, "w.dat");
    	apply_AA(nx, ny, nz, nxd, nzd, W, W_read);
    }
    Alltoall( rank, size, x_jlo, x_jhi, x_klo, x_khi, nxd, ny, nzd, W, w, 1);
    MPI_Barrier(MPI_COMM_WORLD);
    free(W);
    if (rank == 0) transpose_on_rank0( nx, ny, nz, W_read);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Scatterv(W_read, scounts_scat, displs_scat, MPI_DOUBLE, w_read, receive_scat[rank] , MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
    if(rank == 0) free(W_read);
    MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")


  /************************************************ backward FFTs *********************************************/
  if (rank == 0) printf("Reading and Antialiasing completed...\nStarting Backward transformations...\n");
  // ------------------------------------------- Setup z-Transpose --------------------------------------------
  remap3d_create( remap_comm , &remap_zpencil);
  remap3d_setup( remap_zpencil,
      		  	  x_ilo,  x_ihi,  x_jlo, x_jhi,  x_klo,  x_khi,
				  z_ilo,  z_ihi,  z_jlo, z_jhi,  z_klo,  z_khi,
      			  nqty, permute, memoryflag, &sendsize, &recvsize);
  // ----------------------------------------------------------------------------------------------------------
  // Backward FFT#1
  //print_x_pencil(nxd, x_jlo, x_jhi, x_klo, u, rank, scounts[rank], 0);
  MPI_Barrier( MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  double timer_b1 = 0.0;
  timer_b1 -= MPI_Wtime();
  b_FFT_HC2R( u, elem_per_proc, i_length );
  b_FFT_HC2R( v, elem_per_proc, i_length );
  //f_FFT_R2HC( v, elem_per_proc, i_length );
  b_FFT_HC2R( w, elem_per_proc, i_length );
  MPI_Barrier( MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  timer_b1 += MPI_Wtime();
  print_x_pencil(nxd, x_jlo, x_jhi, x_klo, v, rank, scounts[rank], 0);

  // Transpose in z-pencil
  double timer_trasp_z = 0.0, TIMER_TRASP_z = 0.0;
  timer_trasp_z -= MPI_Wtime();
  remap3d_remap(remap_zpencil,u,u,sendbuf,recvbuf);
  remap3d_remap(remap_zpencil,v,v,sendbuf,recvbuf);
  remap3d_remap(remap_zpencil,w,w,sendbuf,recvbuf);
  MPI_Barrier(remap_comm); // @suppress("Symbol is not resolved")
  timer_trasp_z += MPI_Wtime();


  // Backward FFT#2
  double timer_b2 = 0.0;
  timer_b2 -= MPI_Wtime();
 // b_FFT( u, elem_per_proc, k_length );
 // b_FFT( v, elem_per_proc, k_length );
 // b_FFT( w, elem_per_proc, k_length );
  MPI_Barrier( MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  timer_b2 += MPI_Wtime();


  //Finalize plan
  remap3d_destroy(remap_zpencil);
  //print_z_pencil( nz, x_ilo, x_ihi, x_jlo, u, rank, scounts[rank], 0);


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
  //print_z_pencil( nzd, z_ilo, z_ihi, z_jlo, uu, rank, scounts[rank], 1);
  if (rank == 0) printf("Completed\nStarting Forward transformations...\n");

  /************************************************ forward FFTs *********************************************/
  // ---------------------------------------- Setup x-Transpose -----------------------------------------
  remap3d_create( remap_comm , &remap_xpencil);
  permute = 1; 		// From z-contiguous to x-contiguous arrays
  remap3d_setup( remap_xpencil,
		  	  	  z_klo,  z_khi, z_ilo,  z_ihi,  z_jlo, z_jhi,
				  x_klo,  x_khi, x_ilo,  x_ihi,  x_jlo, x_jhi,
				  nqty, permute, memoryflag, &sendsize, &recvsize);
  // -----------------------------------------------------------------------------------------------------------
  // Forward FFT#1
  double timer_f1 = 0.0;
  MPI_Barrier( MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  timer_f1 -= MPI_Wtime();
  f_FFT( u, elem_per_proc, k_length );
  f_FFT( uu, elem_per_proc, k_length );
  f_FFT( uv, elem_per_proc, k_length );
  f_FFT( vv, elem_per_proc, k_length );
  f_FFT( vw, elem_per_proc, k_length );
  f_FFT( ww, elem_per_proc, k_length );
  f_FFT( uw, elem_per_proc, k_length );
  MPI_Barrier( MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  timer_f1 += MPI_Wtime();


  // Transpose to x-pencil
  double timer_trasp_x = 0.0, TIMER_TRASP_x = 0.0;
  timer_trasp_x -= MPI_Wtime();
  remap3d_remap(remap_xpencil,u,u,sendbuf,recvbuf);
  remap3d_remap(remap_xpencil,uu,uu,sendbuf,recvbuf);
  remap3d_remap(remap_xpencil,uv,uv,sendbuf,recvbuf);
  remap3d_remap(remap_xpencil,vv,vv,sendbuf,recvbuf);
  remap3d_remap(remap_xpencil,vw,vw,sendbuf,recvbuf);
  remap3d_remap(remap_xpencil,ww,ww,sendbuf,recvbuf);
  remap3d_remap(remap_xpencil,uw,uw,sendbuf,recvbuf);
  MPI_Barrier(remap_comm); // @suppress("Symbol is not resolved")
  timer_trasp_x += MPI_Wtime();


  // Forward FFT#2
  double timer_f2 = 0.0;
  MPI_Barrier( MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  timer_f2 -= MPI_Wtime();
  f_FFT_R2HC( u, elem_per_proc, i_length );
  f_FFT_R2HC( uu, elem_per_proc, i_length );
  f_FFT( uv, elem_per_proc, i_length );
  f_FFT( vv, elem_per_proc, i_length );
  f_FFT( vw, elem_per_proc, i_length );
  f_FFT( ww, elem_per_proc, i_length );
  f_FFT( uw, elem_per_proc, i_length );
  MPI_Barrier( MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  timer_f2 += MPI_Wtime();

  // Finalize plan
  remap3d_destroy(remap_xpencil);
  free(recvbuf);	 free(sendbuf);

  // De-alias locally
  //print_x_pencil(nxd, x_jlo, x_jhi, x_klo, uu, rank, 2*nxd*(x_jhi-x_jlo+1)*(x_khi-x_klo+1), 2);
  double timer_aax = 0.0;
  timer_aax -= MPI_Wtime();
  x_dealiasing( scounts[rank], modes_per_proc[rank], nx, nxd, u);
  x_dealiasing( scounts[rank], modes_per_proc[rank], nx, nxd, uu);
  x_dealiasing( scounts[rank], modes_per_proc[rank], nx, nxd, uv);
  x_dealiasing( scounts[rank], modes_per_proc[rank], nx, nxd, vv);
  x_dealiasing( scounts[rank], modes_per_proc[rank], nx, nxd, vw);
  x_dealiasing( scounts[rank], modes_per_proc[rank], nx, nxd, ww);
  x_dealiasing( scounts[rank], modes_per_proc[rank], nx, nxd, uw);
  MPI_Barrier( MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  timer_aax += MPI_Wtime();
  if (rank == 0) printf("Completed\nStarting dealiasing operations\n");


  /*********************************************** Modes cutting ********************************************/
  double TIMER_AA = 0.0;
  TIMER_AA -= MPI_Wtime();
  // Alloc memory for the global output
  FFT_SCALAR *UU, *UV, *VV, *VW, *WW, *UW;


  // Gather U data on rank 0
  U = (FFT_SCALAR*) malloc( nx*ny*nzd*2* sizeof(FFT_SCALAR));
  if( U == NULL) {
	  if (rank == 0) {
		  perror(".:Error while allocating gather memory U:.\n");
		  abort();
	  }
  }
  Alltoall( rank, size, x_jlo, x_jhi, x_klo, x_khi, nx, ny, nzd, U, u, -1);
  MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  if (rank == 0) {
  	  z_dealiasing( nx, ny, nz, nxd, nzd, U);
  	  transpose_on_rank0( nx, ny, nz, U);
  }
  else free(U);
  if (rank == 0) {
  	  for (int i  = 0; i < 2*nx*ny*nzd; i++){
  		 // printf("U[%d]= %f\n", i, U[i]);
  	  }
    }

  MPI_Scatterv(U, scounts_scat, displs_scat, MPI_DOUBLE, u, receive_scat[rank] , MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) free(U);
  //print_y_pencil(nx, ny, nz, u, rank, displs_scat[rank], scounts_scat[rank], 0);


  // Gather UU data on rank 0
  UU = (FFT_SCALAR*) malloc( nx*ny*nzd*2* sizeof(FFT_SCALAR));
  if( UU == NULL) {
	  if (rank == 0) {
		  perror(".:Error while allocating gather memory UU:.\n");
  		  abort();
  	  }
  }
  Alltoall( rank, size, x_jlo, x_jhi, x_klo, x_khi, nx, ny, nzd, UU, uu, -1);
  MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  if (rank == 0) {
	  z_dealiasing( nx, ny, nz, nxd, nzd, UU);
	  transpose_on_rank0( nx, ny, nz, UU);
  }
  else free(UU);
  MPI_Scatterv(UU, scounts_scat, displs_scat, MPI_DOUBLE, uu, receive_scat[rank] , MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) free(UU);


  // Gather UV data on rank 0
  UV = (FFT_SCALAR*) malloc( nx*ny*nzd*2* sizeof(FFT_SCALAR));
  if( UV == NULL) {
	  if (rank == 0) {
		  perror(".:Error while allocating gather memory UV:.\n");
		  abort();
	  }
  }
  Alltoall( rank, size, x_jlo, x_jhi, x_klo, x_khi, nx, ny, nzd, UV, uv, -1);
  MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  if (rank == 0) {
	  z_dealiasing( nx, ny, nz, nxd, nzd, UV);
  	  transpose_on_rank0( nx, ny, nz, UV);
  }
  else free(UV);
  MPI_Scatterv(UV, scounts_scat, displs_scat, MPI_DOUBLE, uv, receive_scat[rank] , MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) free(UV);


  // Gather VV data on rank 0
  VV = (FFT_SCALAR*) malloc( nx*ny*nzd*2* sizeof(FFT_SCALAR));
  if( VV == NULL) {
  	  if (rank == 0) {
  		  perror(".:Error while allocating gather memory VV:.\n");
  		  abort();
  	  }
  }
  Alltoall( rank, size, x_jlo, x_jhi, x_klo, x_khi, nx, ny, nzd, VV, vv, -1);
  MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  if (rank == 0) {
	  z_dealiasing( nx, ny, nz, nxd, nzd, VV);
	  transpose_on_rank0( nx, ny, nz, VV);
  }
  else free(VV);
  MPI_Scatterv(VV, scounts_scat, displs_scat, MPI_DOUBLE, vv, receive_scat[rank] , MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) free(VV);


  // Gather VW data on rank 0
  VW = (FFT_SCALAR*) malloc( nx*ny*nzd*2* sizeof(FFT_SCALAR));
  if( VW == NULL) {
	  if (rank == 0) {
		  perror(".:Error while allocating gather memory VW:.\n");
		  abort();
	  }
  }
  Alltoall( rank, size, x_jlo, x_jhi, x_klo, x_khi, nx, ny, nzd, VW, vw, -1);
  MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  if (rank == 0) {
	  z_dealiasing( nx, ny, nz, nxd, nzd, VW);
  	  transpose_on_rank0( nx, ny, nz, VW);
  }
  else free(VW);
  MPI_Scatterv(VW, scounts_scat, displs_scat, MPI_DOUBLE, vw, receive_scat[rank] , MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) free(VW);


  // Gather WW data on rank 0
  WW = (FFT_SCALAR*) malloc( nx*ny*nzd*2* sizeof(FFT_SCALAR));
  if( WW == NULL) {
	  if (rank == 0) {
		  perror(".:Error while allocating gather memory WW:.\n");
		  abort();
	  }
  }
  Alltoall( rank, size, x_jlo, x_jhi, x_klo, x_khi, nx, ny, nzd, WW, ww, -1);
  MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  if (rank == 0) {
	  z_dealiasing( nx, ny, nz, nxd, nzd, WW);
  	  transpose_on_rank0( nx, ny, nz, WW);
  }
  else free(WW);
  MPI_Scatterv(WW, scounts_scat, displs_scat, MPI_DOUBLE, ww, receive_scat[rank] , MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) free(WW);


  // Gather UW data on rank 0
  UW = (FFT_SCALAR*) malloc( nx*ny*nzd*2* sizeof(FFT_SCALAR));
  if( UW == NULL) {
	  if (rank == 0) {
		  perror(".:Error while allocating gather memory UW:.\n");
		  abort();
	  }
  }
  Alltoall( rank, size, x_jlo, x_jhi, x_klo, x_khi, nx, ny, nzd, UW, uw, -1);
  MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  if (rank == 0) {
	  z_dealiasing( nx, ny, nz, nxd, nzd, UW);
	  transpose_on_rank0( nx, ny, nz, UW);
  }
  else free(UW);
  MPI_Scatterv(UW, scounts_scat, displs_scat, MPI_DOUBLE, uw, receive_scat[rank] , MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) free(UW);

  TIMER_AA += MPI_Wtime();
  //print_y_pencil(nx, ny, nz, u, rank, displs_scat[rank], scounts_scat[rank], 3);


  /************************************************ Print Stats *********************************************/
  // Gather all stats
  double TIMER_b1, TIMER_b2, TIMER_f1, TIMER_f2, TIMER_conv, TIMER_AAx;
  MPI_Allreduce(&timer_trasp_x, &TIMER_TRASP_x,1,MPI_DOUBLE,MPI_MAX,remap_comm); // @suppress("Symbol is not resolved")
  MPI_Allreduce(&timer_trasp_z, &TIMER_TRASP_z,1,MPI_DOUBLE,MPI_MAX,remap_comm); // @suppress("Symbol is not resolved")
  MPI_Allreduce(&timer_aax, &TIMER_AAx,1,MPI_DOUBLE,MPI_MAX,remap_comm); // @suppress("Symbol is not resolved")
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
  	  printf("%lgs employed to de-alias locally in x direction\n", TIMER_AAx);
  	  printf("%lgs employed to gather, de-alias in z direction, transpose & scatter data \n", TIMER_AA);
  	  printf("-----------------------------------------------------------\n\n");
  	  printf("Process Ended\n");
  }


  /**************************************** Release Mem & Finalize MPI *************************************/
  free(u);	free(v);	free(w);
  free(uu);	free(uv);	free(vv);	free(vw);	free(ww);	free(uw);

  MPI_Finalize();
}
