/*
 ============================================================================
 Name        : cores_handler.c
 Author      : Mirco Meazzo
 Version     :
 Copyright   : Your copyright notice
 Description : Hello World in C, Ansi-style
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>

void cores_handler( int modes, int size, int elem_per_proc[size]) {
	int rank =0;
	int check=0;

	// Setup matrix
	for (int i = 0; i < size; i++){
		elem_per_proc[i] = 0;
	}
	for (int i = 0; i < modes; i++) {
		elem_per_proc[rank] = elem_per_proc[rank]+1;
		rank = rank+1;

		if (rank == size ) rank =0;
	}
	for (int i = 0; i < size; i++){
	printf("%d modes on rank %d\n", elem_per_proc[i], i);
	check = check+elem_per_proc[i];
	}
	printf("check - modes = %g ", check - modes);

}

int main(void) {
	int nx = 512;
	int modes = (nx+1)*(nx*2+1);
	int processors = 10000;
	int elem_per_proc[processors];
	printf("MODES: %d\n", modes);


	cores_handler( modes, processors, elem_per_proc);

	/*for (int i = 0; i < processors; i++){
		printf("%d modes\n", elem_per_proc[i]);
		}
		*/
}
