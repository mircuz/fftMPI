	/**************************************************************************************************************
	 * 																											  *
	 * 								2D FFT with Pencil Decomposition in MPI Space								  *
	 * 																											  *
	 **************************************************************************************************************
																							Author: Mirco Meazzo */
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <cstdio>
#include <time.h>
#include "remap3d_wrap.h"    // To perform 3D remapping


void print_array( double *work, int insize, int elem_per_proc, int rank, char string[100] );
void FFT( double *c, int N, int isign );
void b_FFT( double *work, int elem_per_proc, int N_trasf);
void f_FFT( double *work, int elem_per_proc, int N_trasf);
void check_results( double *work, double *work_ref, int elem_per_proc);
void generate_inputs(FFT_SCALAR *U, FFT_SCALAR *V, FFT_SCALAR *W, int nfast, int nmid, int nslow, int rank);

#define MODES 10;

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
  int modes, nfast,nmid,nslow;
  modes = MODES;
  int nxd_AA = (modes*3)/2;
  // fftFitting
  int nxd = 4; int nzd = 4;
  while ( nxd < nxd_AA ) {
	  nxd = nxd*2;
  }
  while ( nzd < modes*2+1 ) {
	  nzd = nzd*2;
  }

  // Length of the array along directions
  int i_length = nxd;
  int j_length = modes;
  int k_length = nzd;

  // TOTAL Modes
  nfast = nxd;
  nmid = modes;
  nslow = nzd;

  // Algorithm to factor Nprocs into roughly cube roots
  int npfast,npmid,npslow;
  /*npfast = (int) pow(size,1.0/3.0);
  while (npfast < size) {
    if (size % npfast == 0) break;
    npfast++;
  }
  */
  npfast= 1;
  int npmidslow = size / npfast;
  npmid = (int) sqrt(npmidslow);
  while (npmid < npmidslow) {
    if (npmidslow % npmid == 0) break;
    npmid++;
  }
  npslow = size / npfast / npmid;


  if (rank == 0) {
  	  printf("\n=======================================================\n"
  			  "2D FFT with %dx%dx%d total modes (%d,%d,%d) on %d procs, %dx%dx%d grid\t\n"
  			"=======================================================\n\n",
  			  nfast,nmid,nslow,modes,modes,modes,size,npfast,npmid,npslow);
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
  nslow = modes;

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

  nfast = modes;
  nmid = nxd;
  nslow = nzd;

  // partitioning in y-pencil
  int outy_jlo = (int) 1.0*ipfast*nfast/npfast;						// J fast
  int outy_jhi = (int) 1.0*(ipfast+1)*nfast/npfast - 1;
  int outy_ilo = (int) 1.0*ipmid*nmid/npmid;						// I med
  int outy_ihi = (int) 1.0*(ipmid+1)*nmid/npmid - 1;
  int outy_klo = (int) 1.0*ipslow*nslow/npslow;						// K slow
  int outy_khi = (int) 1.0*(ipslow+1)*nslow/npslow - 1;

  printf("[Y-PENCIL] (j,k,i order)\t"
  		  "On rank %d the coordinates are: "
		  "(%d,%d,%d) -> (%d,%d,%d)\n", rank, outy_ilo, outy_jlo, outy_klo, outy_ihi, outy_jhi, outy_khi );


  void *remap_zpencil, *remap_xpencil, *remap_ypencil;
  int nqty, permute, memoryflag, sendsize, recvsize;
  nqty = 2;			// Use couples of real numbers per grid point
  permute = 2;  		// From x-contiguous to z-contiguous arrays
  memoryflag = 1;		// Self-allocate the buffers


  /******************************************* Size Variables ******************************************/
  int insize = (in_ihi-in_ilo+1) * (in_jhi-in_jlo+1) * (in_khi-in_klo+1);
  int outsize = (out_ihi-out_ilo+1) * (out_jhi-out_jlo+1) * (out_khi-out_klo+1);
  int remapsize = (insize > outsize) ? insize : outsize;
  int elem_per_proc = (nfast*nmid*nslow)*2 /size;

  //printf("pencil dimensions (%d,%d,%d) on rank %d\n", i_length, j_length, k_length, rank);

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


  /********************************************* Memory filling ********************************************/
  // Allocate the arrays
  FFT_SCALAR *V, *U, *W;
  U = (FFT_SCALAR*) malloc( nfast*nmid*nslow*2* sizeof(FFT_SCALAR));
  V = (FFT_SCALAR*) malloc( nfast*nmid*nslow*2* sizeof(FFT_SCALAR));
  W = (FFT_SCALAR*) malloc( nfast*nmid*nslow*2* sizeof(FFT_SCALAR));

  if (rank == 0) {
	  //On rank 0 read the dataset
	  FILE *U_dat;	U_dat = fopen( "u.dat", "r");
	  FILE *V_dat;	V_dat = fopen( "v.dat", "r");
	  FILE *W_dat;	W_dat = fopen( "w.dat", "r");
	  //Fill the allocated arrays
	  for ( int i = 0; i < nfast*nmid*nslow*2; i++) {
		  fscanf( U_dat, "%lf", &U[i]);
		  fscanf( V_dat, "%lf", &V[i]);
		  fscanf( W_dat, "%lf", &W[i]);
		  //printf("I've read %lf\n", U[i]);
	  }
  }
  //Send chunks of array Velocity to all processors
  MPI_Scatter( U, elem_per_proc , MPI_DOUBLE, u, elem_per_proc,  MPI_DOUBLE, 0, MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  MPI_Scatter( V, elem_per_proc , MPI_DOUBLE, v, elem_per_proc,  MPI_DOUBLE, 0, MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  MPI_Scatter( W, elem_per_proc , MPI_DOUBLE, w, elem_per_proc,  MPI_DOUBLE, 0, MPI_COMM_WORLD); // @suppress("Symbol is not resolved")

  //print_array( u, insize, i_length, rank, "Initialized Values");


  /************************************************ backward FFTs *********************************************/
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
  //print_array( u, insize, k_length, rank, "First Transpose of U performed");
  //print_array( w, insize, k_length, rank, "First Transpose of W performed");


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
  //print_array( u, insize, i_length, rank, "Results UU");


  /************************************************ y-Transpose *********************************************/
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

  remap3d_remap(remap_ypencil,uu,uu,sendbuf,recvbuf); 	MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  remap3d_remap(remap_ypencil,uv,uv,sendbuf,recvbuf);	MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  remap3d_remap(remap_ypencil,vv,vv,sendbuf,recvbuf);	MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  remap3d_remap(remap_ypencil,vw,vw,sendbuf,recvbuf);	MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  remap3d_remap(remap_ypencil,ww,ww,sendbuf,recvbuf);	MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  remap3d_remap(remap_ypencil,uw,uw,sendbuf,recvbuf);	MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  timer_trasp_y += MPI_Wtime();


  //Finalize plan
  remap3d_destroy(remap_ypencil);
  //print_array( u, insize, j_length, rank, "y-Transpose of U performed");


  /************************************************ Print Stats *********************************************/
  // Alloc memory for the global output
  FFT_SCALAR *UU, *UV, *VV, *VW, *WW, *UW;
  UU = (FFT_SCALAR*) malloc( nfast*nmid*nslow*2* sizeof(FFT_SCALAR));
  UV = (FFT_SCALAR*) malloc( nfast*nmid*nslow*2* sizeof(FFT_SCALAR));
  VV = (FFT_SCALAR*) malloc( nfast*nmid*nslow*2* sizeof(FFT_SCALAR));
  VW = (FFT_SCALAR*) malloc( nfast*nmid*nslow*2* sizeof(FFT_SCALAR));
  WW = (FFT_SCALAR*) malloc( nfast*nmid*nslow*2* sizeof(FFT_SCALAR));
  UW = (FFT_SCALAR*) malloc( nfast*nmid*nslow*2* sizeof(FFT_SCALAR));

  // Gather all stats
  double TIMER_b1, TIMER_b2, TIMER_f1, TIMER_f2, TIMER_conv;
  MPI_Allreduce(&timer_trasp_x, &TIMER_TRASP_x,1,MPI_DOUBLE,MPI_MAX,remap_comm); // @suppress("Symbol is not resolved")
  MPI_Allreduce(&timer_trasp_z, &TIMER_TRASP_z,1,MPI_DOUBLE,MPI_MAX,remap_comm); // @suppress("Symbol is not resolved")
  MPI_Allreduce(&timer_trasp_y, &TIMER_TRASP_y,1,MPI_DOUBLE,MPI_MAX,remap_comm); // @suppress("Symbol is not resolved")
  MPI_Allreduce(&timer_b1, &TIMER_b1,1,MPI_DOUBLE,MPI_MAX,remap_comm); // @suppress("Symbol is not resolved")
  MPI_Allreduce(&timer_b2, &TIMER_b2,1,MPI_DOUBLE,MPI_MAX,remap_comm); // @suppress("Symbol is not resolved")
  MPI_Allreduce(&timer_f1, &TIMER_f1,1,MPI_DOUBLE,MPI_MAX,remap_comm); // @suppress("Symbol is not resolved")
  MPI_Allreduce(&timer_f2, &TIMER_f2,1,MPI_DOUBLE,MPI_MAX,remap_comm); // @suppress("Symbol is not resolved")
  MPI_Allreduce(&timer_conv, &TIMER_conv,1,MPI_DOUBLE,MPI_MAX,remap_comm); // @suppress("Symbol is not resolved")
  MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
/*  MPI_Gather( uu, elem_per_proc, MPI_DOUBLE, UU, elem_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  MPI_Gather( uv, elem_per_proc, MPI_DOUBLE, UV, elem_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  MPI_Gather( vv, elem_per_proc, MPI_DOUBLE, VV, elem_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  MPI_Gather( vw, elem_per_proc, MPI_DOUBLE, VW, elem_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  MPI_Gather( ww, elem_per_proc, MPI_DOUBLE, WW, elem_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  MPI_Barrier(MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
  MPI_Gather( uw, elem_per_proc, MPI_DOUBLE, UW, elem_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
*/
  // Print stats
  if (rank == 0) {
	  printf("\n-----------------------------------------------------------\n");
	  printf("%lgs employed to perform 2D FFT (backward) \n", TIMER_b1 +TIMER_b2);
	  printf("%lgs employed to transpose the array (z-pencil) \n", TIMER_TRASP_z);
	  printf("%lgs employed to perform 2D FFT (forward) \n", TIMER_f1 +TIMER_f2);
  	  printf("%lgs employed to transpose the array (x-pencil) \n", TIMER_TRASP_x);
  	  printf("%lgs employed to perform convolutions \n", TIMER_conv);
  	  printf("%lgs employed to transpose the array (y-pencil) \n", TIMER_TRASP_y);
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
  free(UU);	free(UV);	free(VV);	free(VW);	free(WW);	free(UW);
  free(recvbuf);
  free(sendbuf);
  MPI_Finalize();
}


/*============================================= Functions Def =============================================*/
void print_array( double *work, int insize, int elem_per_proc, int rank, char string[100] ) {

	//Print work row-wise
	printf("\n\n%s\n"
			"LOCAL array on rank: %d\n", string, rank);
	double re, im;
	int i = 0;
	while ( i < insize*2 ){
		re = work[i];
		i++;
		im = work[i];
		i++;
		printf("%f+i(%f)\t\t", re, im);
		if ( i % (elem_per_proc*2) == 0){
			printf("\n=============\n");
		}
	}
}

void FFT( double *c, int N, int isign ) {
	/**********************************************************************
	  FFT - calculates the discrete fourier transform of an array of double
	  precision complex numbers using the FFT algorithm.

	  c = pointer to an array of size 2*N that contains the real and
	    imaginary parts of the complex numbers. The even numbered indices contain
	    the real parts and the odd numbered indices contain the imaginary parts.
	      c[2*k] = real part of kth data point.
	      c[2*k+1] = imaginary part of kth data point.
	  N = number of data points. The array, c, should contain 2*N elements
	  isign = 1 for forward FFT, -1 for inverse FFT.

	  Copyright (C) 2003, 2004, 2005 Exstrom Laboratories LLC
	*/
  int n, n2, nb, j, k, i0, i1;
  double wr, wi, wrk, wik;
  double d, dr, di, d0r, d0i, d1r, d1i;
  double *cp;

  j = 0;
  n2 = N / 2;
  for( k = 0; k < N; ++k )
  {
    if( k < j )
    {
      i0 = k << 1;
      i1 = j << 1;
      dr = c[i0];
      di = c[i0+1];
      c[i0] = c[i1];
      c[i0+1] = c[i1+1];
      c[i1] = dr;
      c[i1+1] = di;
    }
    n = N >> 1;
    while( (n >= 2) && (j >= n) )
    {
      j -= n;
	  n = n >> 1;
    }
    j += n;
  }

  for( n = 2; n <= N; n = n << 1 )
  {
    wr = cos( 2.0 * M_PI / n );
    wi = sin( 2.0 * M_PI / n );
    if( isign == 1 ) wi = -wi;
    cp = c;
    nb = N / n;
    n2 = n >> 1;
    for( j = 0; j < nb; ++j )
    {
      wrk = 1.0;
      wik = 0.0;
      for( k = 0; k < n2; ++k )
      {
        i0 = k << 1;
        i1 = i0 + n;
        d0r = cp[i0];
        d0i = cp[i0+1];
        d1r = cp[i1];
        d1i = cp[i1+1];
        dr = wrk * d1r - wik * d1i;
        di = wrk * d1i + wik * d1r;
        cp[i0] = d0r + dr;
        cp[i0+1] = d0i + di;
        cp[i1] = d0r - dr;
        cp[i1+1] = d0i - di;
        d = wrk;
        wrk = wr * wrk - wi * wik;
        wik = wr * wik + wi * d;
      }
      cp += n << 1;
    }
  }
}

void b_FFT( double *work, int elem_per_proc, int N_trasf) {

	double* in = (double *) malloc( sizeof(double)*2*N_trasf );
	int count = 0;
	while ( count < elem_per_proc/(2*N_trasf)) {			// To move among rows in the pencil
		// fill IN array
		for ( int i = 0; i < 2*N_trasf; i++ ){
			in[i] = work[i+count*2*N_trasf];

		}
		// Execute FFT & Normalize
		FFT( in, N_trasf, -1 );
		for (int i = 0; i < 2*N_trasf; i++) {
			work[i+count*2*N_trasf] = in[i] / (N_trasf);
		}
		count++;
	}
	free(in);
}

void f_FFT( double *work, int elem_per_proc, int N_trasf) {

	double* in = (double *) malloc( sizeof(double)*2*N_trasf );
	int count = 0;
	while ( count < elem_per_proc/(2*N_trasf)) {			// To move among rows in the pencil
		// fill IN array
		for ( int i = 0; i < 2*N_trasf; i++ ){
			in[i] = work[i+count*2*N_trasf];
		}
		// Execute FFT
		FFT( in, N_trasf, +1 );
		for (int i = 0; i < 2*N_trasf; i++) {
			work[i+count*2*N_trasf] = in[i];

		}
		count++;
	}
	free(in);
}

void check_results( double *work, double *work_ref, int elem_per_proc) {

	/* Remind to add in "Memory Filling" section the following lines od code:
	  FFT_SCALAR *u_ref = (FFT_SCALAR *) malloc(remapsize*sizeof(FFT_SCALAR)*2);
  	  FFT_SCALAR *v_ref = (FFT_SCALAR *) malloc(remapsize*sizeof(FFT_SCALAR)*2);
  	  FFT_SCALAR *w_ref = (FFT_SCALAR *) malloc(remapsize*sizeof(FFT_SCALAR)*2);
  	  MPI_Scatter( U, elem_per_proc , MPI_DOUBLE, u_ref, elem_per_proc,  MPI_DOUBLE, 0, MPI_COMM_WORLD);
  	  MPI_Scatter( V, elem_per_proc , MPI_DOUBLE, v_ref, elem_per_proc,  MPI_DOUBLE, 0, MPI_COMM_WORLD);
  	  MPI_Scatter( W, elem_per_proc , MPI_DOUBLE, w_ref, elem_per_proc,  MPI_DOUBLE, 0, MPI_COMM_WORLD);
 */

	  double mostdiff = 0.0, mydiff= 0.0;
	  for ( int i = 0; i < elem_per_proc; i++) {
		  mydiff = fabs( work[i] - work_ref[i]);
		  if ( mydiff > mostdiff ) {
			  mostdiff = mydiff;
			  printf("Max difference in initial/final values = %.20f\n",mostdiff);
	  	  }
		}
	  //print_array( work_ref, insize, N_trasf, rank, "Reference");
}

void generate_inputs(FFT_SCALAR *U, FFT_SCALAR *V, FFT_SCALAR *W, int nfast, int nmid, int nslow, int rank) {
	//Generate Input
	//int q= 0.0;
	if (rank == 0){
	  for ( int i = 0; i < nslow*nmid*nfast *2; i++) {
		  //U[i] = q++;
		  U[i] = (float)rand()/ (float)(RAND_MAX/10);
	  }
	  for ( int i = 0; i < nslow*nmid*nfast *2; i++) {
		  //V[i] = q++;
		  V[i] = (float)rand()/ (float)(RAND_MAX/10);
	  }
	  for ( int i = 0; i < nslow*nmid*nfast *2; i++) {
		  //W[i] = q++;
		  W[i] = (float)rand()/ (float)(RAND_MAX/10);
	  }
  }

}



