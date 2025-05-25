#ifndef _MMIO_HIGHLEVEL_

#define _MMIO_HIGHLEVEL_



#include "mmio.h"
#include "common.h"

void exclusive_scan(MAT_PTR_TYPE *input, int length);


// read matrix infomation from mtx file

int mmio_info(int *m, int *n, int *nnz, int *isSymmetric, char *filename);



// read matrix infomation from mtx file

int mmio_data(int *csrRowPtr, int *csrColIdx, MAT_VAL_TYPE *csrVal, char *filename);


// read matrix infomation from mtx file
int mmio_allinone(int *m, int *n, MAT_PTR_TYPE *nnz, int *isSymmetric, 
                  MAT_PTR_TYPE **csrRowPtr, int **csrColIdx, MAT_VAL_TYPE **csrVal, 
                  const char *filename);



#endif
