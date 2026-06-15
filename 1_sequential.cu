// sequential N-body simulation - Euler / Velocity Verlet
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

// float return: handle with pointers
void compute_forces(int N, float epsilon, float x[][3], float m[], float F[][3], float *U_out) {
    float U = 0.0, eps2 = epsilon*epsilon; // reset to zero and compute eps2
    for (int i = 0; i < N; i++){
        for (int k = 0; k < 3; k++){
            F[i][k] = 0.0;
        }
    }
    for (int i = 0; i < N; i++){
        for (int j = 0; j < i; j++) {
            float dx = x[i][0]-x[j][0], dy = x[i][1]-x[j][1], dz = x[i][2]-x[j][2]; // distance
            float r2 = dx*dx + dy*dy + dz*dz + eps2; // distance squared (faster than pow)
            float inv_r = 1.0 / sqrt(r2);
            float mimj  = m[i] * m[j];
            float coef  = mimj * inv_r*inv_r*inv_r;
            F[i][0]-=coef*dx; F[j][0]+=coef*dx;  // exploit symmetry
            F[i][1]-=coef*dy; F[j][1]+=coef*dy;
            F[i][2]-=coef*dz; F[j][2]+=coef*dz;
            U -= mimj * inv_r;
        }
    }
    *U_out = U;
}

float gaussian(){
    float u1 = (rand()+1.0) / ((float)RAND_MAX + 2.0); // U(0,1)
    float u2 = (rand()+1.0) / ((float)RAND_MAX + 2.0); // U(0,1)
    return sqrt(-2.0 * log(u1)) * cos(6.2831855307 * u2); // box muller
}

int main(int argc, char **argv) {
    srand(3110);
    //srand(time(NULL));

    int update = 0; // 0 Euler -- 1 Verlet
    int star = 0;   // 0: no star at middle -- 1 star at middle
    int mb = 0;     // 0: random initialization of v[-1,1] -- 1: Maxwell-Boltzmann like distribution
    int N = 100000;  // Num particles
    float T = 2.0; // for maxwell boltzmann distr.
    int custom_ic = 0; // custom initial cond. for plotting trajectories

    // heap allocation (these arrays are far too big for the stack)
    float *m           = (float*)      malloc((size_t)N * sizeof(float));     // masses
    float (*x)[3]      = (float(*)[3]) malloc((size_t)N * 3 * sizeof(float)); // positions
    float (*v)[3]      = (float(*)[3]) malloc((size_t)N * 3 * sizeof(float)); // velocities
    float (*a)[3]      = (float(*)[3]) malloc((size_t)N * 3 * sizeof(float)); // accelerations
    float (*temp_a)[3] = (float(*)[3]) malloc((size_t)N * 3 * sizeof(float)); // storage for old a (Verlet)
    float (*F)[3]      = (float(*)[3]) malloc((size_t)N * 3 * sizeof(float)); // force

    float L = 1000;  // box dim
    float min_dist = 10;  // minimum distance
    bool ensurement = false; //flag
    float dt = 0.02; // time unit
    float dt2 = dt * dt; //computing dt^2 out of the loop saves time
    int iterations = 3; // tot iter
    clock_t start, end; //time measurement
    float diff;        //time measurement
    float U;          // potential energy
    float K;          // kinetic energy
    float epsilon = 0.2; // softening constant to avoid div by zero

    // masses
    for (int i = 0; i < N; i++)
        m[i] = ((float)rand() / (float)(RAND_MAX) * 9) + 1;

    if (star == 1){
        m[0] = 250; 
        printf("star mode activated!\n");
    }

    int it = 0;
    while (it < N) {
        for (int j = 0; j < 3; j++)
            x[it][j] = (float)rand() / (float)RAND_MAX * L;

        // first particle is the star
        if (star == 1 && it == 0){
            for (int j = 0; j < 3; j++){
                x[0][j] = (float)L/2.0;
            }
        }
        ensurement = true;
        for (int j = 0; j < it; j++) {
            float dx = x[it][0]-x[j][0], dy = x[it][1]-x[j][1], dz = x[it][2]-x[j][2];
            float r2 = dx*dx + dy*dy + dz*dz; // distance squared
            if (r2 < min_dist*min_dist) {       // squared vs squared
                ensurement = false;
                break;
            }
        }
        if (ensurement){
            it++;
            printf("placed particle %d\n",it);
        }
    }

    if(mb == 0){
        for (int i = 0; i < N; i++)
            for (int j = 0; j < 3; j++)
                v[i][j] = (float)rand() / (float)RAND_MAX * 2 - 1;  // v in [-1, +1]

        if (star == 1){
            for (int j = 0; j < 3; j++)
                v[0][j] = 0.0;  // star does not move at beginning
        }
    }
    // MAXWELL-BOLTZMANN branch
    if(mb == 1){
        for(int i = 0; i < N; i++){
            float sigma = sqrt(T / m[i]);  // v ~ N(0,kb*T/M)
            for(int j = 0; j<3; j++){
                v[i][j] = sigma * gaussian(); 
            }
        }
    }

    if(custom_ic == 1){
        // masses — set before the IC block
        m[0] = 10.0;
        m[1] = 10.0;

        // positions: separated by 100 along x, centred on 500,500,500
        x[0][0] = 400; x[0][1] = 550; x[0][2] = 550;
        x[1][0] = 600; x[1][1] = 500; x[1][2] = 550;
        
        // v_circ ~ 0.224; use 0.15 to get a clear ellipse (sub-circular)
        v[0][0] = 0.0; v[0][1] =  0.15; v[0][2] = 0.0;
        v[1][0] = 0.0; v[1][1] = -0.15; v[1][2] = 0.0;
    }

    // print initial conditions (first 5 particles)
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
                float temp = 0;
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
                    /*x[i][k] += v[i][k] * dt;
                    v[i][k] += a[i][k] * dt;*/
                     // this order is Euler base 
                    // if revert: energy conservation ~ verlet
                    v[i][k] += a[i][k] * dt;
                    x[i][k] += v[i][k] * dt;
                }
            
            /*
            // Boundary check
            for (int i = 0; i < N; i++)
                for (int k = 0; k < 3; k++) {
                    x[i][k] = fmod(x[i][k], L); // bound in (-L, +L)
                    if (x[i][k] < 0)
                        x[i][k] += L; // bound in (0,+L)
                }
            */

            end = clock();
            diff = ((float) (end-start))/CLOCKS_PER_SEC; //time of iteration

            // diagnostics every 100
            if (diag){
                // compute max acceleration
                float maxa2 = 0;
                float a2 = 0;
                for (int i = 0; i < N; i++) {
                    a2 = 0;
                    for(int k = 0; k < 3; k++){
                        a2 += a[i][k]*a[i][k]; // squared modulo of a
                    }
                    if (a2 > maxa2)
                        maxa2 = a2;
                }
                float maxa = sqrt(maxa2);
                // compute min distance
                float mind = L;
                for (int i = 0; i < N; i++) {
                    for (int j = 0; j < i; j++) {
                        float dx = x[i][0]-x[j][0], dy = x[i][1]-x[j][1], dz = x[i][2]-x[j][2];
                        float d = sqrt(dx*dx + dy*dy + dz*dz); // distance
                        if (d < mind)
                            mind = d;
                    }
                }

                float mint = sqrt(mind / maxa); // bound for dt
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
            for (int i=0;i<N;i++){
                for(int k=0;k<3;k++){
                    x[i][k] += v[i][k]*dt + 0.5*a[i][k]*dt2;
                }
            }
            /*
            // boundary
            for (int i=0;i<N;i++) for(int k=0;k<3;k++) {
                x[i][k] = fmod(x[i][k], L); if (x[i][k] < 0) x[i][k] += L;
            }
            */
            // step 2: recompute forces + a
            memcpy(temp_a, a, (size_t)N * 3 * sizeof(float)); // store a (full array, not sizeof(pointer))
            compute_forces(N, epsilon, x, m, F, &U);
            for (int i=0;i<N;i++){ 
                for(int k=0;k<3;k++){ 
                    a[i][k] = F[i][k]/m[i];
                }
            }

            // step 3: v update
            for (int i=0;i<N;i++){ 
                for(int k=0;k<3;k++){
                    v[i][k] += 0.5*(a[i][k] + temp_a[i][k])*dt;
                }
            }
            end = clock(); diff = ((float)(end-start))/CLOCKS_PER_SEC;

            K = 0;
            for (int i=0;i<N;i++){
                float temp=0;
                for(int k=0;k<3;k++){
                    temp+=v[i][k]*v[i][k];
                }
                K += 0.5*m[i]*temp;
            }

            // diagnostics every 100
            if (diag){
                // compute max acceleration
                float maxa2 = 0;
                float a2 = 0;
                for (int i = 0; i < N; i++) {
                    a2 = 0;
                    for(int k = 0; k < 3; k++){
                        a2 += a[i][k]*a[i][k]; // squared modulo of a
                    }
                    if (a2 > maxa2)
                        maxa2 = a2;
                }
                float maxa = sqrt(maxa2);
                // compute min distance
                float mind = L;
                for (int i = 0; i < N; i++) {
                    for (int j = 0; j < i; j++) {
                        float dx = x[i][0]-x[j][0], dy = x[i][1]-x[j][1], dz = x[i][2]-x[j][2];
                        float d = sqrt(dx*dx + dy*dy + dz*dz); // distance
                        if (d < mind)
                            mind = d;
                    }
                }

                float mint = sqrt(mind / maxa); // bound for dt
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

    free(m); free(x); free(v); free(a); free(temp_a); free(F);
    fclose(fp);
    fclose(tp);
    fclose(ep);
    return 0;
}
