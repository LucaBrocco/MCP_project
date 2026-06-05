// N-body simulation — CUDA GPU version
// Matches physics of directives.cu (softening, Velocity Verlet, periodic BC)
// Compile: nvcc -O2 -arch=native cuda_nbody.cu -o cuda_nbody -lm
//
// Memory layout: flat 1D arrays (GPU doesn't support 2D VLAs).
//   position/velocity/force/acceleration: index as arr[i*3 + k]

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>

// ── helpers ────────────────────────────────────────────────────────────────

// CPU-side distance (used only for initialisation)
static float cpu_dist(float *a, float *b) {
    float d = 0;
    for (int k = 0; k < 3; k++) d += (a[k]-b[k])*(a[k]-b[k]);
    return sqrt(d);
}

// Macro: index into flat [N][3] array
#define IDX(i,k) ((i)*3+(k))
// process CHUNK rows per launch instead of all N
#define CHUNK 64
// ── CUDA error check ────────────────────────────────────────────────────────
#define CUDA_CHECK(call) do {                                        \
    cudaError_t err = (call);                                        \
    if (err != cudaSuccess) {                                        \
        fprintf(stderr, "CUDA error at %s:%d — %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));        \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
} while(0)

// ── Kernels ─────────────────────────────────────────────────────────────────

// Each thread handles one particle i.
// Full N×N loop (Newton's 3rd costs atomic collisions on GPU — full loop is
// cleaner and parallelises perfectly: thread i owns F[i] completely).
__global__ void kernel_forces(int N, float epsilon,
                               float *x, float *m,
                               float *F, float *U_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float Fi0 = 0, Fi1 = 0, Fi2 = 0, Ui = 0;

    for (int j = 0; j < N; j++) {
        if (j == i) continue;
        float dx = x[IDX(i,0)] - x[IDX(j,0)];
        float dy = x[IDX(i,1)] - x[IDX(j,1)];
        float dz = x[IDX(i,2)] - x[IDX(j,2)];
        float r2    = dx*dx + dy*dy + dz*dz + epsilon*epsilon;
        float r     = sqrt(r2);
        float r3    = r2 * r;
        float mimj  = m[i] * m[j];

        Fi0 -= mimj * dx / r3;
        Fi1 -= mimj * dy / r3;
        Fi2 -= mimj * dz / r3;

        // count each pair once
        if (j < i) Ui -= mimj / r;
    }

    F[IDX(i,0)] = Fi0;
    F[IDX(i,1)] = Fi1;
    F[IDX(i,2)] = Fi2;

    // atomic add into global U accumulator
    atomicAdd(U_out, Ui);
}

// Kinetic energy: each thread accumulates K for particle i
__global__ void kernel_kinetic(int N, float *v, float *m, float *K_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float v2 = v[IDX(i,0)]*v[IDX(i,0)]
              + v[IDX(i,1)]*v[IDX(i,1)]
              + v[IDX(i,2)]*v[IDX(i,2)];
    atomicAdd(K_out, 0.5 * m[i] * v2);
}

// a = F / m
__global__ void kernel_accel(int N, float *F, float *m, float *a)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    for (int k = 0; k < 3; k++)
        a[IDX(i,k)] = F[IDX(i,k)] / m[i];
}

// Verlet step 1: x += v*dt + 0.5*a*dt²
__global__ void kernel_verlet_x(int N, float *x, float *v, float *a,
                                 float dt, float dt2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    for (int k = 0; k < 3; k++)
        x[IDX(i,k)] += v[IDX(i,k)]*dt + 0.5*a[IDX(i,k)]*dt2;
}

// Verlet step 3: v += 0.5*(a_new + a_old)*dt
__global__ void kernel_verlet_v(int N, float *v,
                                 float *a_new, float *a_old, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    for (int k = 0; k < 3; k++)
        v[IDX(i,k)] += 0.5 * (a_new[IDX(i,k)] + a_old[IDX(i,k)]) * dt;
}

// Euler update: v += a*dt, x += v*dt
__global__ void kernel_euler(int N, float *x, float *v, float *a, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    for (int k = 0; k < 3; k++) {
        v[IDX(i,k)] += a[IDX(i,k)] * dt;
        x[IDX(i,k)] += v[IDX(i,k)] * dt;
    }
}

// Periodic boundary conditions
__global__ void kernel_boundary_conditions(int N, float *x, float L)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    for (int k = 0; k < 3; k++) {
        x[IDX(i,k)] = fmod(x[IDX(i,k)], L);
        if (x[IDX(i,k)] < 0) x[IDX(i,k)] += L;
    }
}
__global__ void kernel_forces_partial(int N, int i_start, int i_end,
                                      float epsilon, float *x, float *m,
                                      float *F, float *U_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i   = i_start + idx;
    if (i >= i_end) return;

    float Fi0 = 0, Fi1 = 0, Fi2 = 0, Ui = 0;
    for (int j = 0; j < N; j++) {
        if (j == i) continue;
        float dx = x[IDX(i,0)] - x[IDX(j,0)];
        float dy = x[IDX(i,1)] - x[IDX(j,1)];
        float dz = x[IDX(i,2)] - x[IDX(j,2)];
        float r2   = dx*dx + dy*dy + dz*dz + epsilon*epsilon;
        float r    = sqrt(r2);
        float mimj = m[i] * m[j];
        Fi0 -= mimj * dx / (r2*r);
        Fi1 -= mimj * dy / (r2*r);
        Fi2 -= mimj * dz / (r2*r);
        if (j < i) Ui -= mimj / r;
    }
    F[IDX(i,0)] = Fi0;
    F[IDX(i,1)] = Fi1;
    F[IDX(i,2)] = Fi2;
    atomicAdd(U_out, Ui);
}
// ── main ────────────────────────────────────────────────────────────────────

int main(void) {
    srand(time(NULL));

    // ── simulation parameters ──────────────────────────────────────────────
    const int    update     = 1;      // 0 = Euler, 1 = Velocity Verlet
    const int    N          = 50000;
    const float L          = 1000.0;
    const float min_dist   = 10.0;
    const float dt         = 0.03;
    const float dt2        = dt * dt;
    const int    iterations = 100;
    const float epsilon    = 0.5;    // softening
    const int    THREADS    = 256;    // GPU threads per block
    const int    BLOCKS     = (N + THREADS - 1) / THREADS;

    // ── host arrays ────────────────────────────────────────────────────────
    float *h_x      = (float*)malloc(N*3*sizeof(float));
    float *h_v      = (float*)malloc(N*3*sizeof(float));
    float *h_a      = (float*)malloc(N*3*sizeof(float));
    float *h_a_old  = (float*)malloc(N*3*sizeof(float));
    float *h_F      = (float*)malloc(N*3*sizeof(float));
    float *h_m      = (float*)malloc(N*sizeof(float));

    // ── initialisation (CPU) ───────────────────────────────────────────────
    for (int i = 0; i < N; i++)
        h_m[i] = ((float)rand() / RAND_MAX * 9) + 1;

    // place particles ensuring min separation
    int it = 0;
    bool ok;
    while (it < N) {
        for (int k = 0; k < 3; k++)
            h_x[IDX(it,k)] = (float)rand() / RAND_MAX * L;
        ok = true;
        for (int j = 0; j < it; j++)
            if (cpu_dist(&h_x[IDX(it,0)], &h_x[IDX(j,0)]) < min_dist)
                { ok = false; break; }
        if (ok){
            it++;
            printf("spawned particle %d\n",it);
        }
    }

    for (int i = 0; i < N; i++)
        for (int k = 0; k < 3; k++)
            h_v[IDX(i,k)] = (float)rand() / RAND_MAX * 2 - 1;

    // ── device arrays ──────────────────────────────────────────────────────
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

    // ── output files ───────────────────────────────────────────────────────
    FILE *fp = fopen("trajectory.csv", "w");
    FILE *tp = fopen("time.csv",       "w");
    FILE *ep = fopen("energy.csv",     "w");
    if (!fp || !tp || !ep) { fprintf(stderr, "Error opening files\n"); return 1; }
    fprintf(fp, "timestep,particle,x,y,z\n");
    fprintf(tp, "timestep,runtime,maxdt\n");
    fprintf(ep, "timestep,U,K,Etot\n");

    float zero = 0.0;

    // EULER loop
    if (update == 0){
        for (int t = 0; t < iterations; t++) {
            int diag = (t % 100 == 0);
            cudaEvent_t ev_start, ev_stop;
            cudaEventCreate(&ev_start);
            cudaEventCreate(&ev_stop);
            cudaEventRecord(ev_start);

            // 1. Forces + potential energy

            CUDA_CHECK(cudaMemcpy(d_U, &zero, sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_K, &zero, sizeof(float), cudaMemcpyHostToDevice));
            // forces, acceleration, energy, integration, boundary conditions
            kernel_forces<<<BLOCKS,THREADS>>>(N,epsilon, d_x, d_m, d_F, d_U);
            kernel_accel<<<BLOCKS, THREADS>>>(N, d_F, d_m, d_a);
            kernel_kinetic<<<BLOCKS,THREADS>>>(N, d_v, d_m, d_K);
            kernel_euler<<<BLOCKS, THREADS>>>(N, d_x, d_v, d_a, dt);
            kernel_boundary_conditions<<<BLOCKS, THREADS>>>(N, d_x, L);

            // sync and measure time
            cudaEventRecord(ev_stop);
            cudaEventSynchronize(ev_stop);
            float ms;
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
                    float a2 = h_a[IDX(i,0)]*h_a[IDX(i,0)]
                            + h_a[IDX(i,1)]*h_a[IDX(i,1)]
                            + h_a[IDX(i,2)]*h_a[IDX(i,2)];
                    if (a2 > maxa2) maxa2 = a2;
                }
                float maxa = sqrt(maxa2);

                // min distance (CPU)
                float mind = L;
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < i; j++) {
                        float d = cpu_dist(&h_x[IDX(i,0)], &h_x[IDX(j,0)]);
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
    // VERLET loop
    if (update == 1){
        // initial acceleration
        CUDA_CHECK(cudaMemcpy(d_U, &zero, sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, &zero, sizeof(float), cudaMemcpyHostToDevice));
        kernel_forces<<<BLOCKS,THREADS>>>(N,epsilon, d_x, d_m, d_F, d_U);
        kernel_accel<<<BLOCKS, THREADS>>>(N, d_F, d_m, d_a);

        for (int t = 0; t < iterations; t++) {
            int diag = (t % 100 == 0);
            cudaEvent_t ev_start, ev_stop;
            cudaEventCreate(&ev_start);
            cudaEventCreate(&ev_stop);
            cudaEventRecord(ev_start);

            
            // step 1 (first reset energies to zero to allow accumulation)
            CUDA_CHECK(cudaMemcpy(d_a_old, d_a, N*3*sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_U, &zero, sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_K, &zero, sizeof(float), cudaMemcpyHostToDevice));
            kernel_verlet_x<<<BLOCKS, THREADS>>>(N, d_x, d_v, d_a, dt, dt2);
            kernel_boundary_conditions<<<BLOCKS, THREADS>>>(N, d_x, L);

            // step 2 
            kernel_forces<<<BLOCKS,THREADS>>>(N,epsilon, d_x, d_m, d_F, d_U);
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
                    float a2 = h_a[IDX(i,0)]*h_a[IDX(i,0)]
                            + h_a[IDX(i,1)]*h_a[IDX(i,1)]
                            + h_a[IDX(i,2)]*h_a[IDX(i,2)];
                    if (a2 > maxa2) maxa2 = a2;
                }
                float maxa = sqrt(maxa2);

                // min distance (CPU)
                float mind = L;
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < i; j++) {
                        float d = cpu_dist(&h_x[IDX(i,0)], &h_x[IDX(j,0)]);
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