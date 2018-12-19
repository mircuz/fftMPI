/*
 ============================================================================
 Name        : 2d_fft.c
 Author      : Mirco Meazzo
 Version     :
 Copyright   : GNU GPL v. 3
 Description :
 ============================================================================
 */
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <cstdio>
#include <time.h>

#include "remap3d_wrap.h"    // To perform 3D remapping

void print_array( double *work, int insize, int elem_per_fft, int rank, char string[100] ) {

	//Print work row-wise
	printf("\n\n%s\n"
			"LOCAL array on rank: %d\n", string, rank);
	double re, im, *ptr;
	ptr = work;
	int i = 0;
	while ( i < insize*2 ){
		re = work[i];
		i++;
		im = work[i];
		i++;
		printf("%f+i%f\t\t", re, im);
		if ( i % (elem_per_fft*2) == 0){
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
		  "(%d,%d,%d) -> (%d,%d,%d)\n", rank, in_ilo, in_jlo, in_klo, in_ihi, in_jhi, in_khi );

  // partition Output grid into Npfast x Npmid x Npslow
  int out_klo = (int) 1.0*ipfast*nfast/npfast;					// K fast
  int out_khi = (int) 1.0*(ipfast+1)*nfast/npfast - 1;
  int out_ilo = (int) 1.0*ipmid*nmid/npmid;						// I med
  int out_ihi = (int) 1.0*(ipmid+1)*nmid/npmid - 1;
  int out_jlo = (int) 1.0*ipslow*nslow/npslow;					// J slow
  int out_jhi = (int) 1.0*(ipslow+1)*nslow/npslow - 1;

  printf("[AFTER TRANSPOSITION]\t"
		  "On rank %d the coordinates are: "
		  "(%d,%d,%d) -> (%d,%d,%d)\n", rank, out_ilo, out_jlo, out_klo, out_ihi, out_jhi, out_khi );


  //******************************************** Remap Variables ********************************************

  void *remap_backward, *remap_forward;
  remap3d_create( remap_comm , &remap_backward);

  int nqty,permute,memoryflag,sendsize,recvsize;
   nqty = 2;			// Use couples of real numbers per grid point
   	   	   	   	   	    // TODO RIORDINARE da xyz a zxy, quindi cambiare il permute e le coord dei pencil!!
   permute = 2;  		// From x-contiguous to z-contiguous arrays
   memoryflag = 1;		// Self-allocate the buffers

   int insize = (in_ihi-in_ilo+1) * (in_jhi-in_jlo+1) * (in_khi-in_klo+1);
   int outsize = (out_ihi-out_ilo+1) * (out_jhi-out_jlo+1) * (out_khi-out_klo+1);
   //  printf( "%d vs %d is the size on rank %d\n", insize, outsize, rank);
   int remapsize = (insize > outsize) ? insize : outsize;
   FFT_SCALAR *work = (FFT_SCALAR *) malloc(remapsize*sizeof(FFT_SCALAR)*2);
   FFT_SCALAR *sendbuf = (FFT_SCALAR *) malloc(sendsize*sizeof(FFT_SCALAR)*2);
   FFT_SCALAR *recvbuf = (FFT_SCALAR *) malloc(recvsize*sizeof(FFT_SCALAR)*2);


   //********************************************* Memory filling *********************************************
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

   //Send chunks of array V to all processors
   int elem_per_proc = (nfast*nmid*nslow)*2 /size;
   int N_trasf = nfast;
   MPI_Scatter( V, elem_per_proc , MPI_DOUBLE, work, elem_per_proc,  MPI_DOUBLE, 0, MPI_COMM_WORLD); // @suppress("Symbol is not resolved")

   print_array( work, insize, N_trasf, rank, "Initialized Values");


   /*/************************************* Memory filling (IN FFT DATATYPE) ************************************
   //Generate Input

   int q= 0, n_seq = 0;
   fftw_complex *V; // @suppress("Type cannot be resolved")
   V = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nslow*nmid*nfast); // @suppress("Type cannot be resolved")

   if (rank == 0){
	   for (int j = 0; j < nslow; j++)
   		   for ( int k = 0; k < nmid ; k++ )
   			   for ( int i = 0; i < nfast; i++){
   				   V[n_seq][0] = q++;		//L'ordine nella matrice dipende dallo Stride- !!!
   				   V[n_seq][1] = q++;
   				   //printf("FILLING WITH: %f+i%f\n", V[n_seq][0], V[n_seq][1]);
   				   n_seq++;
   			   }
   }

   //Send chunks of array V to all processors
   fftw_complex *work; // @suppress("Type cannot be resolved")
   work = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) *remapsize); // @suppress("Type cannot be resolved")
   int scatter_counter = nfast*nmid*nslow*2 / size;
   MPI_Scatter( V, scatter_counter, MPI_DOUBLE, work, scatter_counter,  MPI_DOUBLE, 0, MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
   print_array( work, insize, 4, rank);	// 4 is the length of the transform


   // Perform FFT along first direction
   double timer_trasf_1D = 0.0;
   timer_trasf_1D -= MPI_Wtime();
   FFT( nfast, nmid, nslow,  work,  rank,  size);
   timer_trasf_1D += MPI_Wtime();
   if (rank == 0)
	   printf("%f sec to perform 1D FFT\n", timer_trasf_1D);
*/


   //************************************************ inverse FFTs **********************************************
   // ---------------------------------------- Setup Backward Transpose -----------------------------------------
     remap3d_setup( remap_backward,
      		  	  in_ilo,  in_ihi,  in_jlo, in_jhi,  in_klo,  in_khi,
                  out_ilo,  out_ihi,  out_jlo, out_jhi,  out_klo,  out_khi,
      			  nqty, permute, memoryflag, &sendsize, &recvsize);
   // -----------------------------------------------------------------------------------------------------------

   // inverse FFT#1		// TODO Creare il ciclo interno ai pencil per effettuare le trasformate
   MPI_Barrier( MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
   clock_t begin_b1 = clock();
  // FFT( work, N_trasf, -1 );
   clock_t end_b1 = clock();
   double time_spent_b1 = (double)(end_b1 - begin_b1) / CLOCKS_PER_SEC;
   printf("%f sec to perform first 1D FFT on rank %d\n", time_spent_b1, rank);


   MPI_Barrier( MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
   double timer_trasp = 0.0, TIMER_TRASP = 0.0;
   timer_trasp -= MPI_Wtime();
   remap3d_remap(remap_backward,work,work,sendbuf,recvbuf);
   timer_trasp += MPI_Wtime();
   MPI_Allreduce(&timer_trasp, &TIMER_TRASP,1,MPI_DOUBLE,MPI_MAX,remap_comm); // @suppress("Symbol is not resolved")
   if (rank == 0) printf("\n%lg sec employed to transpose the array\n", TIMER_TRASP);

   remap3d_destroy(remap_backward);

   // inverse FFT#2
   MPI_Barrier( MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
   clock_t begin_b2 = clock();
  // FFT( work, N_trasf, -1 );
   clock_t end_b2 = clock();
   double time_spent_b2 = (double)(end_b2 - begin_b2) / CLOCKS_PER_SEC;
   printf("%f sec to perform second 1D FFT on rank %d\n", time_spent_b2, rank);

   //Print work transposed row-wise
   //print_array( work, insize, N_trasf, rank, "First Transpose performed);


   //************************************************ inverse FFTs **********************************************
   // ---------------------------------------- Setup Forward Transpose -----------------------------------------
   remap3d_create( remap_comm , &remap_forward);

   permute = 1;
   remap3d_setup( remap_forward,
		   	   	   out_klo,  out_khi, out_ilo,  out_ihi,  out_jlo, out_jhi,
				   in_klo,  in_khi, in_ilo,  in_ihi,  in_jlo, in_jhi,
				   nqty, permute, memoryflag, &sendsize, &recvsize);
   // -----------------------------------------------------------------------------------------------------------

   MPI_Barrier( MPI_COMM_WORLD); // @suppress("Symbol is not resolved")
   remap3d_remap(remap_forward,work,work,sendbuf,recvbuf);
   remap3d_destroy(remap_forward);
   //Print work transposed row-wise
   print_array( work, insize, N_trasf, rank, "Second Transpose Results");



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


  MPI_Finalize();
}
