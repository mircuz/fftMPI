	/**************************************************************************************************************
	 * 																											  *
	 * 							Header file for the functions used into 2d_fft.c 								  *
	 * 																											  *
	 **************************************************************************************************************
																						Author: Dr. Mirco Meazzo */

#ifndef FFT_SUPPORT_H_
#define FFT_SUPPORT_H_


void print_array( double *work, int insize, int elem_per_proc, int rank, char string[100] );
void FFT( double *c, int N, int isign );
void b_FFT( double *work, int elem_per_proc, int N_trasf);
void f_FFT( double *work, int elem_per_proc, int N_trasf);
void dealiasing(int nx, int ny, int nz, int nxd, FFT_SCALAR *U);
void transpose_on_rank0(int nx, int ny, int nz, FFT_SCALAR *U);
void cores_handler( int modes, int size, int modes_per_proc[size]);

// No longer in use
void check_results( double *work, double *work_ref, int elem_per_proc);
void generate_inputs(FFT_SCALAR *U, FFT_SCALAR *V, FFT_SCALAR *W, int nfast, int nmid, int nslow, int rank) ;

#endif /* FFT_SUPPORT_H_ */
