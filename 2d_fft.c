/*
 ============================================================================
 Name        : 2d_fft.c
 Author      : Mirco Meazzo
 Version     :
 Copyright   : Your copyright notice
 Description : Hello MPI World in C 
 ============================================================================
 */
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <cstdio>

#include "fftw3.h"			 // To perform 1D FFT
#include "remap3d_wrap.h"    // To perform 3D remapping


// main program
int main(int narg, char **args) {

  // setup MPI
  MPI_Init(&narg,&args);
  MPI_Comm remap_comm; // @suppress("Type cannot be resolved")
  MPI_Comm_dup( MPI_COMM_WORLD, &remap_comm ); // @suppress("Symbol is not resolved")
  MPI_Comm world = MPI_COMM_WORLD; // @suppress("Symbol is not resolved") // @suppress("Type cannot be resolved")

  int rank,size;
  MPI_Comm_size(world,&size);
  MPI_Comm_rank(world,&rank);

  // Modes
  int nfast,nmid,nslow;
  nfast = 4;
  nmid = 4;
  nslow = 4;

  // Algorithm to factor Nprocs into roughly cube roots
  int npfast,npmid,npslow;

  npfast = (int) pow(size,1.0/3.0);
  while (npfast < size) {
    if (size % npfast == 0) break;
    npfast++;
  }
  int npmidslow = size / npfast;
  npmid = (int) sqrt(npmidslow);
  while (npmid < npmidslow) {
    if (npmidslow % npmid == 0) break;
    npmid++;
  }
  npslow = size / npfast / npmid;

  // partition Input grid into Npfast x Npmid x Npslow
  int ipfast = rank % npfast;
  int ipmid = (rank/npfast) % npmid;
  int ipslow = rank / (npfast*npmid);
  int in_ilo, in_ihi, in_jlo, in_jhi, in_klo, in_khi;

  in_ilo = (int) 1.0*ipfast*nfast/npfast;						// I fast
  in_ihi = (int) 1.0*(ipfast+1)*nfast/npfast - 1;
  in_jlo = (int) 1.0*ipmid*nmid/npmid;							// j med
  in_jhi = (int) 1.0*(ipmid+1)*nmid/npmid - 1;
  in_klo = (int) 1.0*ipslow*nslow/npslow;						// K slow
  in_khi = (int) 1.0*(ipslow+1)*nslow/npslow - 1;

  printf("[BEFORE TRANSPOSITION]\t"
		  "On rank %d the coordinates are: "
		  "(%d,%d,%d) -> (%d,%d,%d)\n\n", rank, in_ilo, in_jlo, in_klo, in_ihi, in_jhi, in_khi );

  // partition Output grid into Npfast x Npmid x Npslow
  int out_klo = (int) 1.0*ipfast*nfast/npfast;					// K fast
  int out_khi = (int) 1.0*(ipfast+1)*nfast/npfast - 1;
  int out_ilo = (int) 1.0*ipmid*nmid/npmid;						// I med
  int out_ihi = (int) 1.0*(ipmid+1)*nmid/npmid - 1;
  int out_jlo = (int) 1.0*ipslow*nslow/npslow;					// J slow
  int out_jhi = (int) 1.0*(ipslow+1)*nslow/npslow - 1;

  printf("[AFTER TRANSPOSITION]\t"
		  "On rank %d the coordinates are: "
		  "(%d,%d,%d) -> (%d,%d,%d)\n\n", rank, out_ilo, out_jlo, out_klo, out_ihi, out_jhi, out_khi );

  // ------------------------------------------- Remap Setup ---------------------------------------------
  void *remap;
  remap3d_create( remap_comm , &remap);

  int nqty,permute,memoryflag,sendsize,recvsize;
   nqty = 2;			// Use couples of real numbers per grid point
   permute = 2;  		// From x-contiguous to z-contiguous arrays
   memoryflag = 1;		// Self-allocate the buffers

  remap3d_setup( remap,
 		  	  in_ilo,  in_ihi,  in_jlo, in_jhi,  in_klo,  in_khi,
               out_ilo,  out_ihi,  out_jlo, out_jhi,  out_klo,  out_khi,
 			  nqty, permute, memoryflag, &sendsize, &recvsize);

   int insize = (in_ihi-in_ilo+1) * (in_jhi-in_jlo+1) * (in_khi-in_klo+1);
   int outsize = (out_ihi-out_ilo+1) * (out_jhi-out_jlo+1) * (out_khi-out_klo+1);
   //  printf( "%d vs %d is the size on rank %d\n", insize, outsize, rank);
   int remapsize = (insize > outsize) ? insize : outsize;
   FFT_SCALAR *work = (FFT_SCALAR *) malloc(remapsize*sizeof(FFT_SCALAR)*2);
   FFT_SCALAR *sendbuf = (FFT_SCALAR *) malloc(sendsize*sizeof(FFT_SCALAR)*2);
   FFT_SCALAR *recvbuf = (FFT_SCALAR *) malloc(recvsize*sizeof(FFT_SCALAR)*2);


   //-------------------------------------------- Memory filling ----------------------------------------
   //Generate Input
   double *p_in;
   p_in = work;
   int q= 0, n_seq = 0;
   FFT_SCALAR *V;
   V = (FFT_SCALAR*) malloc( nfast*nmid*nslow*2* sizeof(FFT_SCALAR));

   if (rank == 0){
	   for (int j = 0; j < nslow; j++)
		   for ( int k = 0; k < nmid ; k++ )
			   for ( int i = 0; i < nfast*2; i++){
				   V[n_seq] = q++;		//L'ordine nella matrice dipende dallo Stride- !!!
				   n_seq++;
			   }
   }
   //TODO fix datatype and elements counter!!
   //Send chunks of array V to all processors
   MPI_Scatter( V, 32, MPI_DOUBLE, work, 32,  MPI_DOUBLE, 0, MPI_COMM_WORLD); // @suppress("Symbol is not resolved")

   //Print work row-wise
   if (rank == 0){
	   printf("\n\nWORK ARRAY\n");
	   double *ptr, A, B;
	   ptr = work;
	   int i = 0, x_plane = 0;
	   while ( i < insize*2 ){
		   A = *ptr;
		   ptr++;
		   i++;
		   B = *ptr;
		   ptr++;
		   i++;
		   printf("%f +i %f \t", A, B);
		   if ( i % 8 == 0){
			   printf("\n=============\n");
			   x_plane++;
		   }
	   }
   }


   // 1D FFT Setup






   MPI_Barrier( MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
   double timer_trasp = 0.0, TIMER_TRASP = 0.0;
   timer_trasp -= MPI_Wtime();
   remap3d_remap(remap,work,work,sendbuf,recvbuf);
   timer_trasp += MPI_Wtime();
   MPI_Allreduce(&timer_trasp, &TIMER_TRASP,1,MPI_DOUBLE,MPI_MAX,remap_comm); // @suppress("Symbol is not resolved")
   if (rank == 0) printf("\n%lg sec employed to transpose the array\n", TIMER_TRASP);

//Print work transposed row-wise
   if (rank == 0){
	   printf("\n\nTRANSPOSED WORK ARRAY\n");
	   double *ptr, A, B;
	   ptr = work;
	   int i = 0, x_plane = 0;
	   while ( i < insize*2 ){
		   A = *ptr;
		   ptr++;
		   i++;
		   B = *ptr;
		   ptr++;
		   i++;
		   printf("%f +i %f \t", A, B);
		   if ( i % 8 == 0) {
			   printf("\n=============\n");
			   x_plane++;
		   }
	   }
   }


 /*  // setup FFT, could replace with tune()

   int fftsize,sendsize,recvsize;
   fft3d_setup(fft,nfast,nmid,nslow,
               ilo,ihi,jlo,jhi,klo,khi,ilo,ihi,jlo,jhi,klo,khi,
               0,&fftsize,&sendsize,&recvsize);

   FFT_SCALAR *work = (FFT_SCALAR *) malloc(2*fftsize*sizeof(FFT_SCALAR));

   int n = 0;
   for (int k = klo; k <= khi; k++) {
     for (int j = jlo; j <= jhi; j++) {
       for (int i = ilo; i <= ihi; i++) {
         work[n] = (double) n;
         n++;
         work[n] = (double) n;
         n++;
       }
     }
   }


   // perform 2 FFTs

   double timestart = MPI_Wtime();
   fft3d_compute(fft,work,work,1);        // forward FFT
   fft3d_compute(fft,work,work,-1);       // inverse FFT
   double timestop = MPI_Wtime();
*/
   if (rank == 0) {
    printf("Two %dx%dx%d FFTs on %d procs as %dx%dx%d grid\n",
            nfast,nmid,nslow,size,npfast,npmid,npslow);
//    printf("CPU time = %g secs\n",timestop-timestart);
   }












   /*
   // find largest difference between initial/final values
   // should be near zero

   n = 0;
   double mydiff = 0.0;
   for (int k = klo; k <= khi; k++) {
     for (int j = jlo; j <= jhi; j++) {
       for (int i = ilo; i <= ihi; i++) {
         if (fabs(work[n]-n) > mydiff) mydiff = fabs(work[n]-n);
         n++;
         if (fabs(work[n]-n) > mydiff) mydiff = fabs(work[n]-n);
         n++;
       }
     }
   }

   double alldiff;
   MPI_Allreduce(&mydiff,&alldiff,1,MPI_DOUBLE,MPI_MAX,world);
   if (rank == 0) printf("Max difference in initial/final values = %g\n",alldiff);
 */


   // clean up



  free(work);
  free(V);
  remap3d_destroy(remap);

  MPI_Finalize();
}
