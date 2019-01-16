	/**************************************************************************************************************
	 * 																											  *
	 * 							Header file for the functions used into 2d_fft.c 								  *
	 * 																											  *
	 **************************************************************************************************************
																						Author: Dr. Mirco Meazzo */

#ifndef FFT_SUPPORT_H_
#define FFT_SUPPORT_H_


void print_array( double *work, int insize, int elem_per_proc, int rank, char string[100] );
void print_y_pencil(int nx, int ny, int nz, FFT_SCALAR *u, int rank,
		int displs, int scounts, int desidered_rank);
void FFT( double *c, int N, int isign );
void b_FFT( double *work, int elem_per_proc, int N_trasf);
void f_FFT( double *work, int elem_per_proc, int N_trasf);
void read_data_and_apply_AA(int nx, int ny, int nz, int nxd, int nzd, FFT_SCALAR *U, FFT_SCALAR *V, FFT_SCALAR *W);
void dealiasing(int nx, int ny, int nz, int nxd, int nzd, FFT_SCALAR *U);
void transpose_on_rank0(int nx, int ny, int nz, FFT_SCALAR U[nx*ny*nz]);
void cores_handler( int modes, int size, int modes_per_proc[size]);

// No longer in use
void check_results( double *work, double *work_ref, int elem_per_proc);
void generate_inputs(FFT_SCALAR *U, FFT_SCALAR *V, FFT_SCALAR *W, int nfast, int nmid, int nslow, int rank) ;

#endif /* FFT_SUPPORT_H_ */