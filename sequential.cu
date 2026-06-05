// sequential N-body simulation - Euler method
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

// double return: handle with pointers
void compute_forces(int N, double epsilon, double x[][3], double m[], double F[][3], double *U_out) {
    double U = 0.0, eps2 = epsilon*epsilon; // reset to zero and compute eps2
    for (int i = 0; i < N; i++){ 
        for (int k = 0; k < 3; k++){
            F[i][k] = 0.0;
        }
    }
    for (int i = 0; i < N; i++){
        for (int j = 0; j < i; j++) {
            double dx = x[i][0]-x[j][0], dy = x[i][1]-x[j][1], dz = x[i][2]-x[j][2]; // distance
            double r2 = dx*dx + dy*dy + dz*dz + eps2; // distance squared (faster than pow)
            double inv_r = 1.0 / sqrt(r2);          
            double mimj  = m[i] * m[j];
            double coef  = mimj * inv_r*inv_r*inv_r; 
            F[i][0]-=coef*dx; F[j][0]+=coef*dx;  // exploit symmetry
            F[i][1]-=coef*dy; F[j][1]+=coef*dy;
            F[i][2]-=coef*dz; F[j][2]+=coef*dz;
            U -= mimj * inv_r;
        }
    }
    *U_out = U;
}

int main(int argc, char **argv) {
    srand(time(NULL));

    int update = 1; // 0 Euler -- 1 Verlet
    int N = 2;  // Num particles
    double m[N];  // masses
    double L = 1000;  // box dim
    double x[N][3];  //positions
    double min_dist = 10;  // minimum distance
    bool ensurement = false; //flag
    double dt = 0.01; // time unit
    double dt2 = dt * dt; //computing dt^2 out of the loop saves time
    int iterations = 10000; // tot iter
    clock_t start, end; //time measurement
    double diff;        //time measurement
    double a[N][3];      // acceleration
    double temp_a[N][3]; // acceleration storage (verlet)
    double F[N][3];    // force
    double U;          // potential energy
    double K;          // kinetic energy
    double epsilon = 0.2; // softening constant to avoid div by zero 
    double v[N][3];   // velocities

    for (int i = 0; i < N; i++)
        m[i] = ((double)rand() / (double)(RAND_MAX) * 9) + 1;

    
    int it = 0;
    while (it < N) {
        for (int j = 0; j < 3; j++)
            x[it][j] = (double)rand() / (double)RAND_MAX * L;

        ensurement = true;                          
        for (int j = 0; j < it; j++) {
            double dx = x[it][0]-x[j][0], dy = x[it][1]-x[j][1], dz = x[it][2]-x[j][2]; 
            double r2 = dx*dx + dy*dy + dz*dz; // distance
            if (r2 < min_dist) {
                ensurement = false;
                break;                              
            }
        }
        if (ensurement)
            it++;
    }

    
    for (int i = 0; i < N; i++)
        for (int j = 0; j < 3; j++)
            v[i][j] = (double)rand() / (double)RAND_MAX * 2 - 1;  // v in [-1, +1]

    // print initial conditions
    printf("initial positions:\n");
    for (int i = 0; i < N; i++) {
        printf("particle %d: ", i);
        for (int j = 0; j < 3; j++)
            printf("%.6f ", x[i][j]);
        printf("\n");
    }
    printf("masses:\n");
    for (int i = 0; i < N; i++)
        printf("particle %d: %f\n", i, m[i]);

    printf("initial velocities:\n");
    for (int i = 0; i < N; i++) {
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

    // Euler loop
    if (update == 0){
        for (int t = 0; t < iterations; t++) {
            int diag = (t % 100 == 0);
            // time measurement
            start = clock();
            // accumulate pairwise forces (Newton's 3rd law)
            // compute potential energy
            compute_forces(N, epsilon, x, m, F, &U);


            // energy computation (t)
            // kinetic
            K = 0;
            for(int i = 0; i<N; i++){
                double temp = 0;
                for(int k=0;k<3;k++){
                    temp += v[i][k] * v[i][k];
                }
                K += m[i] * 0.5 * temp;
            }
            
            // acceleration = F / m
            for (int i = 0; i < N; i++)
                for (int k = 0; k < 3; k++)
                    a[i][k] = F[i][k] / m[i];

            // integration (moving from t to t+1)
            for (int i = 0; i < N; i++)
                for (int k = 0; k < 3; k++) {
                    v[i][k] += a[i][k] * dt;
                    x[i][k] += v[i][k] * dt;
                }

            // Boundary check
            for (int i = 0; i < N; i++)
                for (int k = 0; k < 3; k++) {
                    x[i][k] = fmod(x[i][k], L); // bound in (-L, +L)
                    if (x[i][k] < 0)
                        x[i][k] += L; // bound in (0,+L)
                }

            end = clock();
            diff = ((double) (end-start))/CLOCKS_PER_SEC; //time of iteration

            // diagnostics every 100
            if (diag){
                // compute max acceleration
                double maxa2 = 0;
                double a2 = 0;
                for (int i = 0; i < N; i++) {
                    a2 = 0;
                    for(int k = 0; k < 3; k++){
                        a2 += a[i][k]*a[i][k]; // squared modulo of a          
                    }
                    if (a2 > maxa2)
                        maxa2 = a2;
                }
                double maxa = sqrt(maxa2);
                // compute min distance
                double mind = L;
                double d;
                for (int i = 0; i < N; i++) {
                    for (int j = 0; j < i; j++) {
                        double dx = x[i][0]-x[j][0], dy = x[i][1]-x[j][1], dz = x[i][2]-x[j][2]; 
                        double d = sqrt(dx*dx + dy*dy + dz*dz); // distance
                        if (d < mind)
                            mind = d;
                    }
                }

                double mint = sqrt(mind / maxa); // bound for dt
                printf("max dt:%.10f\n",mint);
            }
            //write to CSV time information
            fprintf(tp, "%d,%.10f\n", t, diff);
            //write to CSV energy information
            fprintf(ep, "%d,%.6f,%.6f,%.6f\n", t, U,K,U+K);

            // write to CSV trajectory (t+1)
            for (int i = 0; i < N; i++)
                fprintf(fp, "%d,%d,%.6f,%.6f,%.6f\n", t, i, x[i][0], x[i][1], x[i][2]);
            printf("%d\n",t);
            }
        }

    // VERLET loop
    if (update == 1) {
        // initial a0
        compute_forces(N, epsilon, x, m, F, &U);
        for (int i=0;i<N;i++){ 
            for(int k=0;k<3;k++){ 
                a[i][k] = F[i][k]/m[i];
            }
        }

        for (int t = 0; t < iterations; t++) {
            int diag = (t % 100 == 0);
            start = clock();

            // step 1: position update
            for (int i=0;i<N;i++) for(int k=0;k<3;k++)
                x[i][k] += v[i][k]*dt + 0.5*a[i][k]*dt2;

            // boundary 
            for (int i=0;i<N;i++) for(int k=0;k<3;k++) {
                x[i][k] = fmod(x[i][k], L); if (x[i][k] < 0) x[i][k] += L;
            }

            // step 2: recompute forces + a
            memcpy(temp_a, a, sizeof(temp_a));
            compute_forces(N, epsilon, x, m, F, &U);   
            for (int i=0;i<N;i++) for(int k=0;k<3;k++) a[i][k] = F[i][k]/m[i];

            // step 3: v update
            for (int i=0;i<N;i++) for(int k=0;k<3;k++)
                v[i][k] += 0.5*(a[i][k] + temp_a[i][k])*dt;

            end = clock(); diff = ((double)(end-start))/CLOCKS_PER_SEC;

            
            K = 0;
            for (int i=0;i<N;i++){ 
                double temp=0; 
                for(int k=0;k<3;k++){
                    temp+=v[i][k]*v[i][k]; 
                }
            K += 0.5*m[i]*temp; 
        }

        // diagnostics every 100
            if (diag){
                // compute max acceleration
                double maxa2 = 0;
                double a2 = 0;
                for (int i = 0; i < N; i++) {
                    a2 = 0;
                    for(int k = 0; k < 3; k++){
                        a2 += a[i][k]*a[i][k]; // squared modulo of a          
                    }
                    if (a2 > maxa2)
                        maxa2 = a2;
                }
                double maxa = sqrt(maxa2);
                // compute min distance
                double mind = L;
                double d;
                for (int i = 0; i < N; i++) {
                    for (int j = 0; j < i; j++) {
                        double dx = x[i][0]-x[j][0], dy = x[i][1]-x[j][1], dz = x[i][2]-x[j][2]; 
                        double d = sqrt(dx*dx + dy*dy + dz*dz); // distance
                        if (d < mind)
                            mind = d;
                    }
                }

                double mint = sqrt(mind / maxa); // bound for dt
                printf("max dt:%.10f\n",mint);
            }
            //write to CSV time information
            fprintf(tp, "%d,%.10f\n", t, diff);
            //write to CSV energy information
            fprintf(ep, "%d,%.6f,%.6f,%.6f\n", t, U,K,U+K);

            // write to CSV
            for (int i = 0; i < N; i++)
                fprintf(fp, "%d,%d,%.6f,%.6f,%.6f\n", t, i, x[i][0], x[i][1], x[i][2]);
            printf("%d\n",t);
            }
        }

            
        



       
    

    fclose(fp);
    fclose(tp);   // add
    fclose(ep);   // add
    return 0;
}