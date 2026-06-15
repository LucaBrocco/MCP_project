#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>

// index flattening
#define IDX(i,k) ((i)*3+(k))
#define THREADS_PER_BLOCK 128
#define CUDA_CHECK(call) do {                                        \
    cudaError_t err = (call);                                        \
    if (err != cudaSuccess) {                                        \
        fprintf(stderr, "CUDA error at %s:%d — %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));        \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
} while(0)

// F kernel: handles particle i avoiding repeated memory accesses
__global__ void kernel_forces(int N, float epsilon, float epsilon2, float *x, float *m, float *F, float *U_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i>= N) return; // memory bound check
    float Fi0 = 0, Fi1 = 0, Fi2 = 0, Ui = 0;

    // load once before j loop
    float xi0 = x[IDX(i,0)];   
    float xi1 = x[IDX(i,1)];
    float xi2 = x[IDX(i,2)];

    // loop performed N-1 times per thread 
    for (int j = 0; j < N; j++) {
        if (j == i) continue;  // self force skip
        float dx = xi0 - x[IDX(j,0)]; // 1 FLOP
        float dy = xi1 - x[IDX(j,1)]; // 1 FLOP
        float dz = xi2 - x[IDX(j,2)]; // 1 FLOP
        float r2    = dx*dx + dy*dy + dz*dz + epsilon2;  // manual distance: faster // 6 FLOPs 
        float inv_r  = rsqrtf(r2); // 1 FLOP (not really counted though -> SFU)
        float inv_r3 = inv_r * inv_r * inv_r; // 3 FLOPs
        float mimj  = m[i] * m[j]; // 1 FLOP
        Fi0 -= mimj * dx * inv_r3; // 3 FLOPs
        Fi1 -= mimj * dy * inv_r3; // 3 FLOPs
        Fi2 -= mimj * dz * inv_r3; // 3 FLOPs
        // do not count repeated
        if (j < i){ 
            Ui -= mimj * inv_r; // 2 FLOPs but half iter; consider as 1
        }
    }
    // TOTAL FLOPs 1 loop: 23
    // TOTAL: 23 * (N-1) per thread
    F[IDX(i,0)] = Fi0;
    F[IDX(i,1)] = Fi1;
    F[IDX(i,2)] = Fi2;
    // atomic add into global U 
    atomicAdd(U_out, Ui);
}

// K: accumulate onto i particle
__global__ void kernel_kinetic(int N, float *v, float *m, float *K_out)
{
    // total FLOPs: 5 + 1 + 1 + 1= 8
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float v2 = v[IDX(i,0)]*v[IDX(i,0)] + v[IDX(i,1)]*v[IDX(i,1)] + v[IDX(i,2)]*v[IDX(i,2)];
    atomicAdd(K_out, 0.5f * m[i] * v2);
}

// a: loop across i
__global__ void kernel_accel(int N, float *F, float *m, float *a)
{   
    // total FLOPs: 3
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    for (int k = 0; k < 3; k++)
        a[IDX(i,k)] = F[IDX(i,k)] / m[i];
}

// VERLET step 1
__global__ void kernel_verlet_x(int N, float *x, float *v, float *a, float dt, float dt2)
{
    // total FLOPs: 3 * 4 = 12
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    for (int k = 0; k < 3; k++){
        x[IDX(i,k)] += v[IDX(i,k)]*dt + 0.5f*a[IDX(i,k)]*dt2;
    }
}

// VERLET step 3
__global__ void kernel_verlet_v(int N, float *v, float *a_new, float *a_old, float dt)
{
    // total FLOPs: 3 * 4 = 12
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    for (int k = 0; k < 3; k++){
        v[IDX(i,k)] += 0.5 * (a_new[IDX(i,k)] + a_old[IDX(i,k)]) * dt;
    }
}

// EULER integration step
__global__ void kernel_euler(int N, float *x, float *v, float *a, float dt)
{
    // total FLOPs: 3 * (2+2) = 12
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    for (int k = 0; k < 3; k++) {
        v[IDX(i,k)] += a[IDX(i,k)] * dt;
        x[IDX(i,k)] += v[IDX(i,k)] * dt; // EULER-CROMER
    }
}

// bcs
__global__ void kernel_boundary_conditions(int N, float *x, float L)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    for (int k = 0; k < 3; k++) {
        x[IDX(i,k)] = fmod(x[IDX(i,k)], L);
        if (x[IDX(i,k)] < 0) x[IDX(i,k)] += L;
    }
}
// for initialization
float gaussian(){
    float u1 = (rand()+1.0) / ((float)RAND_MAX + 2.0); // U(0,1)
    float u2 = (rand()+1.0) / ((float)RAND_MAX + 2.0); // U(0,1)
    return sqrt(-2.0 * log(u1)) * cos(6.2831855307 * u2); 
}

int main(void) {
    srand(3110);

    int update = 0; // 0 Euler -- 1 Verlet
    int star = 0;   // 0: no star at middle -- 1 star at middle
    int mb = 0;     // 0: random initialization of v[-1,1] -- 1: Maxwell-Boltzmann like distribution

    int iterations = 10; // tot iter
    int N = 50000; // Num particles
    float T = 2.0; // for maxwell boltzmann distr.
    int custom_ic = 0; // custom initial cond. for plotting trajectories

    float L = 1000;  // box dim
    float min_dist = 10;  // minimum distance
    bool ensurement = false; //flag
    float dt = 0.01; // time unit
    float dt2 = dt * dt; //computing dt^2 out of the loop saves time
    clock_t start, end; //time measurement
    float diff;        //time measurement
    float U;          // potential energy
    float K;          // kinetic energy
    float epsilon = 0.2; // softening constant to avoid div by zero
    float epsilon2 = epsilon * epsilon;

    const int    THREADS    = THREADS_PER_BLOCK;    // GPU threads per block
    const int    BLOCKS     = (N + THREADS - 1) / THREADS; // total blocks to launch for N
    
    float *h_x      = (float*)malloc(N*3*sizeof(float));
    float *h_v      = (float*)malloc(N*3*sizeof(float));
    float *h_a      = (float*)malloc(N*3*sizeof(float));
    float *h_a_old  = (float*)malloc(N*3*sizeof(float));
    float *h_F      = (float*)malloc(N*3*sizeof(float));
    float *h_m      = (float*)malloc(N*sizeof(float));

    // initialization (CPU)
    // masses
    for (int i = 0; i < N; i++)
        h_m[i] = ((float)rand() / (float)(RAND_MAX) * 9) + 1;

    if (star == 1){
        h_m[0] = 250; 
        printf("star mode activated!\n");
    }

    int it = 0;
    while (it < N) {
        for (int j = 0; j < 3; j++)
            h_x[IDX(it,j)] = (float)rand() / (float)RAND_MAX * L;

        // first particle is the star
        if (star == 1 && it == 0){
            for (int j = 0; j < 3; j++){
                h_x[IDX(0,j)] = (float)L/2.0;
            }
        }
        ensurement = true;
        if (N<12000){ // run this only for small N
            for (int j = 0; j < it; j++) {
                float dx = h_x[IDX(it,0)]-h_x[IDX(j,0)], dy = h_x[IDX(it,1)]-h_x[IDX(j,1)], dz = h_x[IDX(it,2)]-h_x[IDX(j,2)];
                float r2 = dx*dx + dy*dy + dz*dz; // distance squared
                if (r2 < min_dist*min_dist) {       // squared vs squared
                    ensurement = false;
                    break;
                }
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
                h_v[IDX(i,j)] = (float)rand() / (float)RAND_MAX * 2 - 1;  // v in [-1, +1]

        if (star == 1){
            for (int j = 0; j < 3; j++)
                h_v[IDX(0,j)] = 0.0;  // star does not move at beginning
        }
    }
    // MAXWELL-BOLTZMANN branch
    if(mb == 1){
        for(int i = 0; i < N; i++){
            float sigma = sqrt(T / h_m[i]);  // v ~ N(0,kb*T/M)
            for(int j = 0; j<3; j++){
                h_v[IDX(i,j)] = sigma * gaussian(); 
            }
        }
    }

    if(custom_ic == 1){
        // masses 
        h_m[0] = 10.0;
        h_m[1] = 10.0;

        // positions
        h_x[IDX(0,0)] = 400; h_x[IDX(0,1)] = 550; h_x[IDX(0,2)] = 550;
        h_x[IDX(1,0)] = 600; h_x[IDX(1,1)] = 500; h_x[IDX(1,2)] = 550;
        
        //v 
        h_v[IDX(0,0)] = 0.0; h_v[IDX(0,1)] =  0.15; h_v[IDX(0,2)] = 0.0;
        h_v[IDX(1,0)] = 0.0; h_v[IDX(1,1)] = -0.15; h_v[IDX(1,2)] = 0.0;
    }
    
    // device memory
    float *d_x, *d_v, *d_a, *d_a_old, *d_F, *d_m, *d_U, *d_K;
    CUDA_CHECK(cudaMalloc(&d_x,     N*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v,     N*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a,     N*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a_old, N*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_F,     N*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m,     N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_U,     sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K,     sizeof(float)));

    // copy initial state to GPU
    CUDA_CHECK(cudaMemcpy(d_x, h_x, N*3*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v, N*3*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_m, h_m, N*sizeof(float),   cudaMemcpyHostToDevice));

    FILE *fp = fopen("trajectory.csv", "w");
    FILE *tp = fopen("time.csv",       "w");
    FILE *ep = fopen("energy.csv",     "w");
    if (!fp || !tp || !ep) { fprintf(stderr, "Error opening files\n"); return 1; }
    fprintf(fp, "timestep,particle,x,y,z\n");
    fprintf(tp, "timestep,runtime,kernelruntime\n");
    fprintf(ep, "timestep,U,K,Etot\n");

    float zero = 0.0;

    // EULER loop
    if (update == 0){
        for (int t = 0; t < iterations; t++) {
            int diag = (t % 100 == 0 && t > 0);
            float ms, kms;
            cudaEvent_t ev_start, ev_stop, kernel_start, kernel_stop;
            cudaEventCreate(&ev_start);
            cudaEventCreate(&ev_stop);
            cudaEventCreate(&kernel_start);
            cudaEventCreate(&kernel_stop);
            cudaEventRecord(ev_start);
            
            //forces + potential energy

            CUDA_CHECK(cudaMemcpy(d_U, &zero, sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_K, &zero, sizeof(float), cudaMemcpyHostToDevice));


            // forces, acceleration, energy, integration, boundary conditions
            cudaEventRecord(kernel_start);
            kernel_forces<<<BLOCKS,THREADS>>>(N, epsilon, epsilon2, d_x, d_m, d_F, d_U);
            CUDA_CHECK(cudaGetLastError());
            cudaEventRecord(kernel_stop);
            cudaDeviceSynchronize();
            cudaEventSynchronize(kernel_stop);
            kernel_accel<<<BLOCKS, THREADS>>>(N, d_F, d_m, d_a);
            kernel_kinetic<<<BLOCKS,THREADS>>>(N, d_v, d_m, d_K);
            kernel_euler<<<BLOCKS, THREADS>>>(N, d_x, d_v, d_a, dt);
            cudaEventElapsedTime(&kms, kernel_start, kernel_stop);
            cudaEventDestroy(kernel_start);
            cudaEventDestroy(kernel_stop);

            //kernel_boundary_conditions<<<BLOCKS, THREADS>>>(N, d_x, L);

            // sync and measure time
            cudaEventRecord(ev_stop);
            cudaEventSynchronize(ev_stop);
            cudaEventElapsedTime(&ms, ev_start, ev_stop);
            cudaEventDestroy(ev_start);
            cudaEventDestroy(ev_stop);

            float U, K;
            CUDA_CHECK(cudaMemcpy(&U, d_U, sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&K, d_K, sizeof(float), cudaMemcpyDeviceToHost));

            // diagnostics
            if(diag){
                // if diagnostics we copy back
                CUDA_CHECK(cudaMemcpy(h_x, d_x, N*3*sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_a, d_a, N*3*sizeof(float), cudaMemcpyDeviceToHost));
                // max acceleration (CPU)
                float maxa2 = 0;
                for (int i = 0; i < N; i++) {
                    float a2 = h_a[IDX(i,0)]*h_a[IDX(i,0)] + h_a[IDX(i,1)]*h_a[IDX(i,1)] + h_a[IDX(i,2)]*h_a[IDX(i,2)];
                    if (a2 > maxa2) maxa2 = a2;
                }
                float maxa = sqrt(maxa2);

                // min distance (CPU)
                float mind = L;
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < i; j++) {
                        float dx = h_x[IDX(i,0)] - h_x[IDX(j,0)], dy = h_x[IDX(i,1)] - h_x[IDX(j,1)], dz = h_x[IDX(i,2)] - h_x[IDX(j,2)];
                        float d = dx * dx + dy * dy + dz * dz;
                        if (d < mind) mind = d;
                    }
                float mint = sqrt(mind / maxa);
                printf("max dt:%.10f\n",mint);
                printf("%d\n",t);
            }
            //write to CSV time information
            fprintf(tp, "%d,%.10f,%.10f\n", t, ms/1000.0f, kms/1000.0f);
            //write to CSV energy information
            fprintf(ep, "%d,%.6f,%.6f,%.6f\n", t, U, K, U + K);
        }
    }
    // VERLET loop
    if (update == 1){
        // initial acceleration
        CUDA_CHECK(cudaMemcpy(d_U, &zero, sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, &zero, sizeof(float), cudaMemcpyHostToDevice));
        kernel_forces<<<BLOCKS,THREADS>>>(N,epsilon, epsilon2, d_x, d_m, d_F, d_U);
        kernel_accel<<<BLOCKS, THREADS>>>(N, d_F, d_m, d_a);

        for (int t = 0; t < iterations; t++) {
            int diag = (t % 100 == 0);
            cudaEvent_t ev_start, ev_stop;
            cudaEventCreate(&ev_start);
            cudaEventCreate(&ev_stop);
            cudaEventRecord(ev_start);

            
            // step 1 
            CUDA_CHECK(cudaMemcpy(d_a_old, d_a, N*3*sizeof(float), cudaMemcpyDeviceToDevice)); // copy a
            CUDA_CHECK(cudaMemcpy(d_U, &zero, sizeof(float), cudaMemcpyHostToDevice)); // U = 0
            CUDA_CHECK(cudaMemcpy(d_K, &zero, sizeof(float), cudaMemcpyHostToDevice)); // K = 0
            kernel_verlet_x<<<BLOCKS, THREADS>>>(N, d_x, d_v, d_a, dt, dt2);
            //kernel_boundary_conditions<<<BLOCKS, THREADS>>>(N, d_x, L);

            // step 2 
            kernel_forces<<<BLOCKS,THREADS>>>(N, epsilon, epsilon2, d_x, d_m, d_F, d_U);
            kernel_accel<<<BLOCKS, THREADS>>>(N, d_F, d_m, d_a);

            // step 3
            kernel_verlet_v<<<BLOCKS, THREADS>>>(N, d_v, d_a, d_a_old, dt);

            kernel_kinetic<<<BLOCKS,THREADS>>>(N, d_v, d_m, d_K);
            
            // sync and measure time
            cudaEventRecord(ev_stop);
            cudaEventSynchronize(ev_stop);
            float ms;
            cudaEventElapsedTime(&ms, ev_start, ev_stop);
            cudaEventDestroy(ev_start);
            cudaEventDestroy(ev_stop);

            // copy back to save
            float U, K;
            CUDA_CHECK(cudaMemcpy(&U, d_U, sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&K, d_K, sizeof(float), cudaMemcpyDeviceToHost));

            // diagnostics
            if(diag){
                // if diagnostics we need to copy
                CUDA_CHECK(cudaMemcpy(h_x, d_x, N*3*sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_a, d_a, N*3*sizeof(float), cudaMemcpyDeviceToHost));
                // max acceleration (CPU)
                float maxa2 = 0;
                for (int i = 0; i < N; i++) {
                    float a2 = h_a[IDX(i,0)]*h_a[IDX(i,0)] + h_a[IDX(i,1)]*h_a[IDX(i,1)] + h_a[IDX(i,2)]*h_a[IDX(i,2)];
                    if (a2 > maxa2) maxa2 = a2;
                }
                float maxa = sqrt(maxa2);

                // min distance (CPU)
                float mind = L;
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < i; j++) {
                        float dx = h_x[IDX(i,0)] - h_x[IDX(j,0)], dy = h_x[IDX(i,1)] - h_x[IDX(j,1)], dz = h_x[IDX(i,2)] - h_x[IDX(j,2)];
                        float d = dx * dx + dy * dy + dz * dz;
                        if (d < mind) mind = d;
                    }
                float mint = sqrt(mind / maxa);
                printf("max dt:%.10f\n",mint);
                printf("%d\n",t);
            }
            //write to CSV time information
            fprintf(tp, "%d,%.10f\n", t, ms/1000.0f);
            //write to CSV energy information
            fprintf(ep, "%d,%.6f,%.6f,%.6f\n", t, U, K, U + K);
        }
    }

    fclose(fp); 
    fclose(tp); 
    fclose(ep);

    // cleanup
    free(h_x); free(h_v); free(h_a); free(h_a_old); free(h_F); free(h_m);
    cudaFree(d_x); cudaFree(d_v); cudaFree(d_a); cudaFree(d_a_old);
    cudaFree(d_F); cudaFree(d_m); cudaFree(d_U); cudaFree(d_K);
    return 0;
}
