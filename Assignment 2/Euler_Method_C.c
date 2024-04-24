#include <stdio.h>
#include <math.h>

double exact_solution(double t) {
    return pow(t + 1, 2) - 0.5 * exp(t);
}

double f(double t, double y) {
    return y - pow(t, 2) + 1;
}

double euler_method(double t0, double y0, double h, double T) {
    double t = t0;
    double y = y0;
    while (t < T) {
        double error_bound = 0.5 * h *  (0.5*exp(2)-2)* (exp(t)-1); // Error bound derived in class
        double exact = exact_solution(t);
        double error = fabs(exact - y);
        printf("t = %.2f, y = %.6f, exact = %.6f, error = %.6f, error bound = %.6f\n", t, y, exact, error, error_bound);
        y = y + h * f(t, y);
        t = t + h;
    }
    return y;
}

int main() {
    double t0 = 0.0;
    double y0 = 0.5;
    double h = 0.2;
    double T = 2.0;
    printf("Using Euler's method with h = %.1f\n", h);
    euler_method(t0, y0, h, T);
    return 0;
}
