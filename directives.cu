// GPU-parallel version of the code, with directives
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <omp.h>

// total cores
#define NUM_THREADS 2

// double return: handle with pointers
// basic parallelization strategy (static): each thread acts onto a single row of F: we lose arithmetic advantage given by 3rd Newton's law, but ensures threads independence and avoids race conditions
static void compute_forces(int N,double epsilon, double x[][3], double m[], double F[][3], double *U_out)
{
    double U = 0.0, eps2 = epsilon*epsilon; // reset to zero and compute eps2

    // loop on i: rows of F (each thread computes over a particle)
    #pragma omp parallel for reduction(+:U) schedule(static) num_threads(NUM_THREADS)
    for (int i = 0; i < N; i++){ 
        for (int k = 0; k < 3; k++){
            F[i][k] = 0.0;  // reset forces to 0
        }
        // full j to N: no symmetry exploitation
        for (int j = 0; j < N; j++) {
            if (i == j) continue;
            double dx=x[i][0]-x[j][0], dy=x[i][1]-x[j][1], dz=x[i][2]-x[j][2];
            double r2 = dx*dx + dy*dy + dz*dz + eps2;
            double inv_r = 1.0/sqrt(r2);
            double mimj  = m[i]*m[j];
            double coef  = mimj*inv_r*inv_r*inv_r;
            F[i][0]-=coef*dx; F[i][1]-=coef*dy; F[i][2]-=coef*dz;   
            
            U -= mimj*inv_r;  
                                   
        }
        
    }
    *U_out = U / 2; // we double counted
}

int main(int argc, char **argv) {
    omp_set_num_threads(NUM_THREADS); // default num threads
    srand(time(NULL));

    int update = 1; // 0 Euler -- 1 Verlet
    int N = 10000;  // Num particles

    double L = 1000;  // box dim
    double min_dist = 10;  // minimum distance
    bool ensurement = false; //flag
    double dt = 0.01; // time unit
    double dt2 = dt * dt; //computing dt^2 out of the loop saves time
    int iterations = 1000; // tot iter
    double diff;        //time measurement
    double U;          // potential energy
    double K;          // kinetic energy
    double epsilon = 0.5; // softening constant to avoid div by zero 

    double (*x)[3]      = (double(*)[3]) malloc(N * 3 * sizeof(double));
    double (*v)[3]      = (double(*)[3]) malloc(N * 3 * sizeof(double));
    double (*a)[3]      = (double(*)[3]) malloc(N * 3 * sizeof(double));
    double (*temp_a)[3] = (double(*)[3]) malloc(N * 3 * sizeof(double));
    double (*F)[3]      = (double(*)[3]) malloc(N * 3 * sizeof(double));
    double *m           = (double*)      malloc(N * sizeof(double));

    // mass assignment
    for (int i = 0; i < N; i++)
        m[i] = ((double)rand() / (double)(RAND_MAX) * 9) + 1;

    // position assignment
    int it = 0;
    while (it < N) {
        for (int j = 0; j < 3; j++)
            x[it][j] = (double)rand() / (double)RAND_MAX * L;

        ensurement = true;
        for (int k = 0; k < it; k++) {                                     
            double dx = x[it][0]-x[k][0], dy = x[it][1]-x[k][1], dz = x[it][2]-x[k][2];
            double r2 = dx*dx + dy*dy + dz*dz;
            if (r2 < min_dist*min_dist) { ensurement = false; break; }        // squared vs squared
        }
        if (ensurement) it++;
    }

    // speed assignment
    for (int i = 0; i < N; i++)
        for (int j = 0; j < 3; j++)
            v[i][j] = (double)rand() / (double)RAND_MAX * 2 - 1;  // v in [-1, +1]

    // print initial conditions (5 particles)
    printf("initial positions:\n");
    for (int i = 0; i < 5; i++) {
        printf("particle %d: ", i);
        for (int j = 0; j < 3; j++)
            printf("%.6f ", x[i][j]);
        printf("\n");
    }
    printf("masses:\n");
    for (int i = 0; i < 5; i++)
        printf("particle %d: %f\n", i, m[i]);

    printf("initial velocities:\n");
    for (int i = 0; i < 5; i++) {
        printf("particle %d: ", i);
        for (int j = 0; j < 3; j++)
            printf("%.6f ", v[i][j]);
        printf("\n");
    }

    FILE *fp = fopen("trajectory.csv", "w");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file\n");
        return 1;
    }
    FILE *tp = fopen("time.csv", "w");
    if (tp == NULL) {
        fprintf(stderr, "Error opening file\n");
        return 1;
    }
    FILE *ep = fopen("energy.csv", "w");
    if (ep == NULL) {
        fprintf(stderr, "Error opening file\n");
        return 1;
    }
    fprintf(fp, "timestep,particle,x,y,z\n");
    fprintf(tp, "timestep,runtime,maxdt\n");
    fprintf(ep, "timestep,U,K,Etot\n");

    // EULER LOOP 
    if (update == 0){
        for (int t = 0; t < iterations; t++) {
                int diag = (t % 100 == 0);
                // time measurement (use this rather than clock)
                double wt_start = omp_get_wtime();
                
                // accumulate pairwise forces (Newton's 3rd law)
                // compute potential energy
                compute_forces(N, epsilon, x, m, F, &U);

                // energy computation (t)
                // kinetic
                K = 0;
                // parallelization over i
                #pragma omp parallel for reduction(+:K) schedule(static)
                for(int i = 0; i<N; i++){
                    double temp = 0;
                    for(int k=0;k<3;k++){
                        temp += v[i][k] * v[i][k];
                    }
                    K += m[i] * 0.5 * temp;
                }
            
                // acceleration = F / m ++ integration t+1 ++ boundary check
                // parallelization over i
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < N; i++)
                    for (int k = 0; k < 3; k++) {
                        a[i][k]  = F[i][k] / m[i];
                        v[i][k] += a[i][k] * dt;
                        x[i][k] += v[i][k] * dt;
                        x[i][k]  = fmod(x[i][k], L); if (x[i][k] < 0) x[i][k] += L;
                    }

            
                // clock would show all cores time 
                // this way diff is time of one iter 
                diff = omp_get_wtime() - wt_start;

                // diagnostics
                if(diag){
                    // compute max acceleration
                    double maxa2 = 0;
                    double a2 = 0;
                    // parallelize over i
                    #pragma omp parallel for reduction(max:maxa2) schedule(static)
                    for (int i = 0; i < N; i++) {
                        a2 = a[i][0]*a[i][0] + a[i][1]*a[i][1] + a[i][2]*a[i][2]; // squared modulo of a
                        if (a2 > maxa2)
                            maxa2 = a2;
                    }
                    double maxa = sqrt(maxa2);
                    // compute min distance
                    double mind = L;
                    double d;
                    // parallelize over i
                    #pragma omp parallel for reduction(min:mind) schedule(static)
                    for (int i = 0; i < N; i++) {
                        for (int j = 0; j < i; j++) {
                            double dx=x[i][0]-x[j][0], dy=x[i][1]-x[j][1], dz=x[i][2]-x[j][2];
                            double d = sqrt(dx*dx + dy*dy + dz*dz);
                            if (d < mind)
                                mind = d;
                        }
                    }

                    double mint = sqrt(mind / maxa); // bound for dt
                    printf("max dt:%.10f\n",mint);
                    printf("%d\n",t);
                }
            //write to CSV time information
            fprintf(tp, "%d,%.10f\n", t, diff);
            //write to CSV energy information
            fprintf(ep, "%d,%.6f,%.6f,%.6f\n", t, U,K,U+K);

            // write to CSV
            for (int i = 0; i < N; i++)
                fprintf(fp, "%d,%d,%.6f,%.6f,%.6f\n", t, i, x[i][0], x[i][1], x[i][2]);
            //printf("%d\n",t);
        }
    }

    // VERLET LOOP 
    if (update == 1){
        // initial a0
        compute_forces(N, epsilon, x, m, F, &U);
        // acceleration = F / m
        // parallelization over i
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++)
            for (int k = 0; k < 3; k++)
                a[i][k] = F[i][k] / m[i];

        for (int t = 0; t < iterations; t++) {
                int diag = (t % 100 == 0);
                // time measurement (use this rather than clock)
                double wt_start = omp_get_wtime();
                
                // integration step
                memcpy(temp_a, a, (size_t)N*3*sizeof(a)); // store a
                // parallelization over i
                #pragma omp parallel for schedule(static)
                // 1: position update
                for(int i=0;i<N;i++){
                    for(int k=0;k<3;k++){
                        x[i][k] += v[i][k]*dt + 0.5 * a[i][k]*dt2;
                    }
                }
                // boundary check
                // parallelization over i
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < N; i++)
                    for (int k = 0; k < 3; k++) {
                        x[i][k] = fmod(x[i][k], L); // bound in (-L, +L)
                        if (x[i][k] < 0)
                            x[i][k] += L; // bound in (0,+L)
                    }
                // 2: new acceleration 
                // middle step
                compute_forces(N, epsilon, x, m, F, &U);
                // acceleration
                // parallelization over i 
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < N; i++)
                    for (int k = 0; k < 3; k++)
                        a[i][k] = F[i][k] / m[i];
                // 3: velocity update
                // parallelization over i
                #pragma omp parallel for schedule(static)
                for(int i = 0; i<N; i++){
                    for(int k=0;k<3;k++){
                        v[i][k] += 0.5 * (a[i][k] + temp_a[i][k]) * dt;
                    }
                }
                diff = omp_get_wtime() - wt_start;

                // energy computation (t+1)
                // kinetic
                K = 0;
                // parallelization over i
                #pragma omp parallel for reduction(+:K) schedule(static)
                for(int i = 0; i<N; i++){
                    double temp = 0;
                    for(int k=0;k<3;k++){
                        temp += v[i][k] * v[i][k];
                    }
                    K += m[i] * 0.5 * temp;
                }
        
                // diagnostics
                if(diag){
                    // compute max acceleration
                    double maxa2 = 0;
                    double a2 = 0;
                    // parallelize over i
                    #pragma omp parallel for reduction(max:maxa2) schedule(static)
                    for (int i = 0; i < N; i++) {
                        a2 = a[i][0]*a[i][0] + a[i][1]*a[i][1] + a[i][2]*a[i][2]; // squared modulo of a
                        if (a2 > maxa2)
                            maxa2 = a2;
                    }
                    double maxa = sqrt(maxa2);
                    // compute min distance
                    double mind = L;
                    double d;
                    // parallelize over i
                    #pragma omp parallel for reduction(min:mind) schedule(static)
                    for (int i = 0; i < N; i++) {
                        for (int j = 0; j < i; j++) {
                            double dx=x[i][0]-x[j][0], dy=x[i][1]-x[j][1], dz=x[i][2]-x[j][2];
                            double d = sqrt(dx*dx + dy*dy + dz*dz);
                            if (d < mind)
                                mind = d;
                        }
                    }

                    double mint = sqrt(mind / maxa); // bound for dt
                    printf("max dt:%.10f\n",mint);
                    printf("%d\n",t);
                }
            //write to CSV time information
            fprintf(tp, "%d,%.10f\n", t, diff);
            //write to CSV energy information
            fprintf(ep, "%d,%.6f,%.6f,%.6f\n", t, U,K,U+K);

            // write to CSV
            for (int i = 0; i < N; i++)
                fprintf(fp, "%d,%d,%.6f,%.6f,%.6f\n", t, i, x[i][0], x[i][1], x[i][2]);
           //printf("%d\n",t);
        }
    }


    fclose(fp);
    fclose(tp);   // add
    fclose(ep);   // add
    return 0;
}