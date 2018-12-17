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

#include "fft3d_wrap.h"
#include "remap3d_wrap.h"    // if performing Remaps

// FFT size

#define NFAST 4
#define NMID 4
#define NSLOW 4

// precision-dependent settings
int precision = 2;

// main program

int main(int narg, char **args)
{
  // setup MPI

  MPI_Init(&narg,&args);
  MPI_Comm remap_comm; // @suppress("Type cannot be resolved")
  MPI_Comm_dup( MPI_COMM_WORLD, &remap_comm ); // @suppress("Symbol is not resolved")
  MPI_Comm world = MPI_COMM_WORLD; // @suppress("Symbol is not resolved") // @suppress("Type cannot be resolved")

  int rank,size;
  MPI_Comm_size(world,&size);
  MPI_Comm_rank(world,&rank);

  // instantiate FFT

  void *fft;
  fft3d_create(world,precision,&fft);

  // simple algorithm to factor Nprocs into roughly cube roots

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

  // partition grid into Npfast x Npmid x Npslow bricks

  int nfast,nmid,nslow;
  int in_ilo, in_ihi, in_jlo, in_jhi, in_klo, in_khi;

  nfast = NFAST;
  nmid = NMID;
  nslow = NSLOW;

  int ipfast = rank % npfast;
  int ipmid = (rank/npfast) % npmid;
  int ipslow = rank / (npfast*npmid);

  in_ilo = (int) 1.0*ipfast*nfast/npfast;
  in_ihi = (int) 1.0*(ipfast+1)*nfast/npfast - 1;
  in_jlo = (int) 1.0*ipmid*nmid/npmid;
  in_jhi = (int) 1.0*(ipmid+1)*nmid/npmid - 1;
  in_klo = (int) 1.0*ipslow*nslow/npslow;
  in_khi = (int) 1.0*(ipslow+1)*nslow/npslow - 1;

  printf("BEFORE TRANSPOSITION\n"
		  "On rank %d the coordinates are: "
		  "(%d,%d,%d) -> (%d,%d,%d)\n\n", rank, in_ilo, in_jlo, in_klo, in_ihi, in_jhi, in_khi );


  int out_klo = (int) 1.0*ipfast*nfast/npfast;
  int out_khi = (int) 1.0*(ipfast+1)*nfast/npfast - 1;
  int out_jlo = (int) 1.0*ipmid*nmid/npmid;
  int out_jhi = (int) 1.0*(ipmid+1)*nmid/npmid - 1;
  int out_ilo = (int) 1.0*ipslow*nslow/npslow;
  int out_ihi = (int) 1.0*(ipslow+1)*nslow/npslow - 1;

  printf("AFTER TRANSPOSITION\n"
		  "On rank %d the coordinates are: "
		  "(%d,%d,%d) -> (%d,%d,%d)\n\n", rank, out_ilo, out_jlo, out_klo, out_ihi, out_jhi, out_khi );


  void *remap;
  remap3d_create( remap_comm , &remap);




  int nqty,permute,memoryflag,sendsize,recvsize;
   nqty = 1;			// Trasformation use couple of complex values
   permute = 0;  		// MID index become FAST index and vice-versa
   memoryflag = 1;		// Self-allocate the buffers


  remap3d_setup( remap,
 		  	  in_ilo,  in_ihi,  in_jlo,
               in_jhi,  in_klo,  in_khi,
               out_ilo,  out_ihi,  out_jlo,
               out_jhi,  out_klo,  out_khi,
 			  nqty, permute, memoryflag, &sendsize, &recvsize);



   int insize = (in_ihi-in_ilo+1) * (in_jhi-in_jlo+1) * (in_khi-in_klo+1);
   int outsize = (out_ihi-out_ilo+1) * (out_jhi-out_jlo+1) * (out_khi-out_klo+1);
   printf( "%d vs %d is the size on rank %d\n", insize, outsize, rank);

   int remapsize = (insize > outsize) ? insize : outsize;
   FFT_SCALAR *work = (FFT_SCALAR *) malloc(remapsize*sizeof(FFT_SCALAR)*2);
   FFT_SCALAR *sendbuf = (FFT_SCALAR *) malloc(sendsize*sizeof(FFT_SCALAR)*2);
   FFT_SCALAR *recvbuf = (FFT_SCALAR *) malloc(recvsize*sizeof(FFT_SCALAR)*2);







   //-------------------------------------------- Memory filling ----------------------------------------
   //Generate Input
     double *p_in;
     p_in = work;
     int q= 0, n_seq = 0;
     double Vmat[4][4][8];
     FFT_SCALAR *V;
     V = (FFT_SCALAR*) malloc( nfast*nmid*nslow*2* sizeof(FFT_SCALAR));


     for (int j = 0; j < 4; j++)
    	 for ( int k = 0; k < 4 ; k++ )
    		 for ( int i = 0; i < 4; i++){
        	  Vmat[i][j][k] = q++;

        	  V[n_seq] = Vmat[i][j][k];
        	  n_seq++;
          }






     /*
     for ( int i = 0; i < nfast*nmid*nslow*2 ; i++ ) {
   	  V[i] = j++;
   	  //V[i] = rand() % 10;
     }
     */

     //Send chunks of array V to all processors
     MPI_Scatter( V, 32, MPI_DOUBLE, work, 32,  MPI_DOUBLE, 0, MPI_COMM_WORLD); // @suppress("Symbol is not resolved")


     /*/ Print V
     if (rank == 0){
     double *ptr, A, B;
       ptr = V;
       for ( int y = 0; y < nslow; y++)
     	  for (int  x = 0; x < nmid; x++)
     		  for ( int z = 0; z < nfast; z++){
     			  A = *ptr;
     			  ptr++;
     			  B = *ptr;
     			  ptr++;
     			  printf("R: %f \t I: %f \n", A, B);
     		  }
     }
*/

     //Print work row-wise
          if (rank == 0){
        	  double *ptr, A, B;
        	  ptr = work;
        	  int i = 0;
        	  while ( i < insize*2 ){
        		  A = *ptr;
        		  ptr++;
        		  i++;
        		  B = *ptr;
        		  ptr++;
        		  i++;
        		  printf("%f +i %f \t", A, B);
        		  if ( i % 8 == 0)
        			  printf("\n---------------\n");
        	  }
          }


     /*/Print work
     if (rank == 2){
          double *ptr, A, B;
            ptr = work;
            int i = 0;
            while ( i < insize*2 ){

            	A = *ptr;
            	ptr++;
            	i++;
            	B = *ptr;
            	ptr++;
            	i++;
            	printf("Rw: %f \t\t Iw: %f \n", A, B);
            	if ( i % 8 == 0)
            		printf("---------------\n");
          		  }
          }
*/

  remap3d_remap(remap,work,work,sendbuf,recvbuf);

  //Print work transposed row-wise
         if (rank == 0){
              double *ptr, A, B;
                ptr = work;
                int i = 0;
                while ( i < insize*2 ){
              	  A = *ptr;
              	  ptr++;
              	  i++;
              	  B = *ptr;
              	  ptr++;
              	  i++;
              	  printf("%f +i %f \t", A, B);
              	  if ( i % 8 == 0)
              		  printf("\n---------------\n");
              		  }
              }


  /*/Print work transposed
       if (rank == 3){
            double *ptr, A, B;
              ptr = work;
              int i = 0;
              while ( i < insize*2 ){
            	  A = *ptr;
            	  ptr++;
            	  i++;
            	  B = *ptr;
            	  ptr++;
            	  i++;
            	  printf("Rw_t: %f \t\t Iw_t: %f \n", A, B);
            	  if ( i % 8 == 0)
            		  printf("---------------\n");
            		  }
            }
*/


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
    printf("CPU time = %g secs\n",timestop-timestart);
   }

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

  // ~Remap3d();                                // destructor
 //  free(work);


  remap3d_destroy(remap);
  fft3d_destroy(fft);
  MPI_Finalize();
}
