#include <stdio.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>


void lu_decomposition(double A[], int size, double L[], double U[]) {
    gsl_matrix_view gsl_A = gsl_matrix_view_array(A, size, size);
    gsl_permutation *p = gsl_permutation_alloc(size);
    int signum;


    gsl_linalg_LU_decomp(&gsl_A.matrix, p, &signum);


    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (i > j)
                L[i * size + j] = gsl_matrix_get(&gsl_A.matrix, i, j);
            else if (i == j)
                L[i * size + j] = 1.0;
            else
                L[i * size + j] = 0.0;


            if (i <= j)
                U[i * size + j] = gsl_matrix_get(&gsl_A.matrix, i, j);
            else
                U[i * size + j] = 0.0;
        }
    }


    gsl_permutation_free(p);
}


int main() {
    // First matrix
    double A[] = {3, -1, 1, 3, 6, 2, 3, 3, 7};
    double B[] = {10, -1, 0, -1, 10, -2, 0, -2, 10};
    double C[] = {10, 5, 0, 0, 5, 10, -4, 0, 0, -4, 8, -1, 0, 0, -3, 5};
    double D[] = {4, 1, 1, 0, 1, -1, -3, 1, 1, 0, 2, 1, 5, -1, -1, -1, -1 -1, 4, 0, 0, 2, -1, 1, 4};
    int size = 5;


    // Allocate space for L and U matrices
    double LA[3 * 3];
    double UA[3 * 3];
    double LB[3 * 3];
    double UB[3 * 3];
    double LC[4 * 4];
    double UC[4 * 4];
    double LD[5 * 5];
    double UD[5 * 5];

    // Perform LU decomposition
    lu_decomposition(A, 3, LA, UA);
    lu_decomposition(B, 3, LB, UB);
    lu_decomposition(C, 4, LC, UC);
    lu_decomposition(D, 5, LD, UD);

    // Print the results
    printf("Matrix L for first matrix:\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%8.7f ", LA[i * 3 + j]);
        }
        printf("\n");
    }


    printf("Matrix U for first matrix:\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%8.7f ", UA[i * 3 + j]);
        }
        printf("\n");
    }

    printf("Matrix L for second matrix:\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%8.7f ", LB[i * 3 + j]);
        }
        printf("\n");
    }

    printf("Matrix U for second matrix:\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%8.7f ", UB[i * 3 + j]);
        }
        printf("\n");
    }

    printf("Matrix L for third matrix:\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%8.7f ", LC[i * 4 + j]);
        }
        printf("\n");
    }


    printf("Matrix U for third matrix:\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%8.7f ", UC[i * 4 + j]);
        }
        printf("\n");
    }

    printf("Matrix L for fourth matrix:\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%8.7f ", LD[i * 5 + j]);
        }
        printf("\n");
    }


    printf("Matrix U for fourth matrix:\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%8.7f ", UD[i * 5 + j]);
        }
        printf("\n");
    }
    
    // Checking that the LU decomposition is correct
    // Create GSL matrix views of LA and UA
    gsl_matrix_view LAM = gsl_matrix_view_array(LA, 3, 3);
    gsl_matrix_view UAM = gsl_matrix_view_array(UA, 3, 3);

    // Create a result matrix RA
    gsl_matrix *RA = gsl_matrix_alloc(3, 3);

    // Perform matrix multiplication
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &LAM.matrix, &UAM.matrix, 0.0, RA);

    // Output the result matrix RA
    printf("LU=A check for first matrix:\n");
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            printf("%8.3f ", gsl_matrix_get(RA, i, j));
        }
        printf("\n");
    }

    // Free the memory allocated for matrix RA
    gsl_matrix_free(RA);

    // Create GSL matrix views of LB and UB
    gsl_matrix_view LBM = gsl_matrix_view_array(LB, 3, 3);
    gsl_matrix_view UBM = gsl_matrix_view_array(UB, 3, 3);

    // Create a result matrix RB
    gsl_matrix *RB = gsl_matrix_alloc(3, 3);

    // Perform matrix multiplication
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &LBM.matrix, &UBM.matrix, 0.0, RB);

    // Output the result matrix RB
    printf("LU=A check for second matrix:\n");
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            printf("%8.3f ", gsl_matrix_get(RB, i, j));
        }
        printf("\n");
    }

    // Free the memory allocated for matrix RB
    gsl_matrix_free(RB);

    // Create GSL matrix views of LC and UC
    gsl_matrix_view LCM = gsl_matrix_view_array(LC, 4, 4);
    gsl_matrix_view UCM = gsl_matrix_view_array(UC, 4, 4);

    // Create a result matrix RC
    gsl_matrix *RC = gsl_matrix_alloc(4, 4);

    // Perform matrix multiplication
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &LCM.matrix, &UCM.matrix, 0.0, RC);

    // Output the result matrix RC
    printf("LU=A check for third matrix:\n");
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            printf("%8.3f ", gsl_matrix_get(RC, i, j));
        }
        printf("\n");
    }

    // Free the memory allocated for matrix RC
    gsl_matrix_free(RC);

    // Create GSL matrix views of LC and UC
    gsl_matrix_view LDM = gsl_matrix_view_array(LD, 5, 5);
    gsl_matrix_view UDM = gsl_matrix_view_array(UD, 5, 5);

    // Create a result matrix RC
    gsl_matrix *RD = gsl_matrix_alloc(5, 5);

    // Perform matrix multiplication
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &LDM.matrix, &UDM.matrix, 0.0, RD);

    // Output the result matrix RD
    printf("LU=A check for fourth matrix:\n");
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            printf("%8.3f ", gsl_matrix_get(RD, i, j));
        }
        printf("\n");
    }

    // Free the memory allocated for matrix RD
    gsl_matrix_free(RD);

    return 0;
}

