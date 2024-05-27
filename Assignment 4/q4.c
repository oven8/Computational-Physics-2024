#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

double exp_distribution(double alpha) {
    double x = ((double) rand() / (RAND_MAX));
    return -log(1 - x) / alpha;
}

int main() {
    const int n = 10000;
    const double alpha = 2.0;
    double random[n];

    srand((unsigned) time(NULL));
    for (int i = 0; i < n; i++) {
        random[i] = exp_distribution(alpha);
    }

    FILE *file = fopen("q4_data.csv","w");
    if (file == NULL) {
        printf("Error Opening File!\n");
        return 1;
    }

    for (int i = 0; i < n; i++) {
        fprintf(file, "%f\n", random[i]);
    }

    fclose(file);

    return 0;
}