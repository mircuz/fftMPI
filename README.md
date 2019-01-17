# 2D FFT on a 3D array with pencil decomposition

## Requirements
- fftMPI Library by Sandia National Laboratories (Release 1 Oct 2018)

## Description
This script does a pencil decomposition and perform a 2D FFT along x and z dimensions
on a complex dataset, with whatever number of modes and processors.

Finally the array is transposed in order to get y-pencils.
FFTs & transpositions have timers to show how many time is spent to perform
such actions.
Number of modes is selected during the declarations of the variables at top of
the program.

## How does it works
First of all the script load a complex dataset, apply antialiasing and setup the first transpose.
Once this is done it does a 1D FFT along X, transpose the data across the domain
and than perform another 1D FFT, this time along Z direction.
At this point convolutions are performed.
Then the transpose, for the convolved arrays, is setted up.
Once again we performe a 1D FFT, followed by transpose and another 1D FFT.

The final y transposition is performed plane-by-plane on the processor 0, since all the data are already available there for the dealiasing operations.

The results check function is implemented, but turned off since the
convolutions take place. To use it, just copy the portions of code following the
instructions and turn off the convolutions.

I provide no warranty about the results of the convolutions.


Feel free to modify everything, I do not give warranty for anything!


Author: Mirco Meazzo
