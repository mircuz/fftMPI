	/**************************************************************************************************************
	 * 																											  *
	 * 						Functions developed in order to perform 2D FFT in MPI								  *
	 * 																											  *
	 **************************************************************************************************************
																						Author: Dr. Mirco Meazzo */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef double FFT_SCALAR;

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

void dealiasing(int nx, int ny, int nz, int nxd, FFT_SCALAR *U) {
int i, stride_y, stride_z, reader=0, last_index;
	  for ( stride_z = 0; stride_z < nz*ny*nxd*2; stride_z = stride_z + ny*nxd*2) {
   		  //printf("\n\nstride z %d\n", stride_z );
   	  	  for ( stride_y = 0; stride_y < ny*nxd*2; stride_y = stride_y + nxd*2) {
   	  		//printf("\nstride y %d\n", stride_y );
   			  for ( i = 0; i < (nx)*2; i++) {

   	  			  U[reader] = U[stride_z + stride_y+i];
   	  			  //printf("U[%d] =  %g\n", (reader), U[reader]);
   	  			  reader++;
   	  		  }
   	  	  }
   	  	  last_index = stride_z + stride_y;
	  }
}

void transpose_on_rank0(int nx, int ny, int nz, FFT_SCALAR *U) {
	struct cmplx {
		double re, im;
	};

	int reader = 0, writer = 0;
	// Fill the array on rank 0
	cmplx u_mat[nx][ny];
	for (int k = 0; k < nz; k++) {

		// Read the k-th plane
		for (int j = 0; j < ny; j++) {
			for (int i = 0; i < nx; i++) {
				//printf("U[%d] = %g\n", reader, U[reader]);
				u_mat[i][j].re = U[reader];
				reader++;
				//printf("U[%d] = %g\n", reader, U[reader]);
				u_mat[i][j].im = U[reader];
				reader++;
			}
		}
		// Transpose the k-th plane
	  for (int i = 0; i < nx; i++) {
		  for (int j = 0; j < ny; j++) {
			  U[writer] = u_mat[i][j].re;
			  //printf("U[%d] = %g\n", writer, U[writer]);
			  writer++;
			  U[writer] = u_mat[i][j].im;
			  //printf("U[%d] = %g\n", writer, U[writer]);
			  writer++;
  		  }
  	  }
	}

}

void cores_handler( int modes, int size, int modes_per_proc[size]) {
	int rank =0;
	int check=0;

	for (int i = 0; i < modes; i++) {
		modes_per_proc[rank] = modes_per_proc[rank]+1;
		rank = rank+1;
		if (rank == size ) rank = 0;
	}
	
	for (int i = 0; i < size; i++){
		//printf("%d modes on rank %d\n", modes_per_proc[i], i);
		check = check+modes_per_proc[i];
	}
	if ( (int)(check - modes) != 0 ) {
			printf("[ERROR] check - modes = %d!!\nUnable to scatter modes properly\nAbort... \n", check - modes);
	}


}

void print_y_pencil(int nx, int ny, int nz, FFT_SCALAR *u, int rank,
		int displs, int scounts, int desidered_rank) {
if (rank == desidered_rank) {
	  int total_modes = displs/ (ny*2);
	  int stride_nz = total_modes / nx;
	  int stride_nx = total_modes - stride_nz * nx;

	  for (int i = 0; i < scounts; i++) {
   	  if ( i % (ny*2) == 0) {
   		  printf("========(nx= %d, nz= %d)=======\n", stride_nx , stride_nz);
   		  stride_nx ++;
   		  if ( (stride_nx ) % nx == 0) {
   			  stride_nx =0;
   			  stride_nz ++;
   		  }
   	  }
   	 printf("u[%d]= %g\n", (i), u[i]);
     }
 }
}

void read_data_and_apply_AA(int nx, int ny, int nz, int nxd, int nzd, FFT_SCALAR *U, FFT_SCALAR *V, FFT_SCALAR *W) {

	// Allocate the arrays
		  FFT_SCALAR  *U_read, *V_read, *W_read;
	  	  U_read = (FFT_SCALAR*) malloc( nx*ny*nz*2* sizeof(FFT_SCALAR));
	  	  V_read = (FFT_SCALAR*) malloc( nx*ny*nz*2* sizeof(FFT_SCALAR));
	  	  W_read = (FFT_SCALAR*) malloc( nx*ny*nz*2* sizeof(FFT_SCALAR));

	  	  //On rank 0 read the dataset
	  	  FILE *U_dat;	U_dat = fopen( "u.dat", "r");
	  	  FILE *V_dat;	V_dat = fopen( "v.dat", "r");
	  	  FILE *W_dat;	W_dat = fopen( "w.dat", "r");
	  	  for ( int i = 0; i < (nx)*(ny)*(nz)*2; i++) {
	  		  fscanf( U_dat, "%lf", &U_read[i]);
	  		  fscanf( V_dat, "%lf", &V_read[i]);
	  		  fscanf( W_dat, "%lf", &W_read[i]);
	  		  //printf("I've read %lf\n", U_read[i]);
	  	  }

	  	  //Fill the array with read values and zeros for AA
	  	  int i, stride_y, stride_z, reader=0, last_index;
	  	  for ( stride_z = 0; stride_z < nz*ny*nxd*2; stride_z = stride_z + ny*nxd*2) {
	  		  //printf("\n\nstride z %d\n", stride_z );
	  	  	  for ( stride_y = 0; stride_y < ny*nxd*2; stride_y = stride_y + nxd*2) {
	  	  		//printf("\nstride y %d\n", stride_y );
	  			  for ( i = 0; i < (nx)*2; i++) {
	  	  			  U[stride_z + stride_y+i] = U_read[reader];
	  	  			  V[stride_z + stride_y+i] = V_read[reader];
	  	  			  W[stride_z + stride_y+i] = W_read[reader];
	  	  			  //printf("U[%d] =  %g\n", (stride_z + stride_y+i), U[stride_z + stride_y+i]);
	  	  			  reader++;
	  	  		  }
	  	  		  for ( i = (nx)*2; i < nxd*2; i++) {
	  	  			  U[stride_z + stride_y+i] = 0;
	  	  			  V[stride_z + stride_y+i] = 0;
	  	  			  W[stride_z + stride_y+i] = 0;
	  	  			  //printf("U[%d] =  %g\n", (stride_z + stride_y+i), U[stride_z + stride_y+i]);
	  	  		  }
	  	  	  }
	  	  	  last_index = stride_z + stride_y;
	  	  }
	  	  //Fill with zeros from nz to nzd
	  	  for ( int i = last_index; i < nzd*nxd*ny*2; i++) {
	  		  U[i] = 0;
	  		  V[i] = 0;
	  		  W[i] = 0;
	  	  }
	  	  free(U_read);		free(V_read);		free(W_read);
}
