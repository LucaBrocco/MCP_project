// sequential N-body simulation - Euler method
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <float.h>

#define TILE_SIZE 256
#define IDX(i,k) ((i)*3+(k))
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d — %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// max float in atomic fashion
__device__ void atomicMax_float(float *addr, float val) {
    int *addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
}
// min float
__device__ void atomicMin_float(float *addr, float val) {
    int *addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
                        __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

float distance(float *x1, float *x2) {
    float d = 0;
    for (int i = 0; i < 3; i++)
        d += (x1[i]-x2[i])*(x1[i]-x2[i]);
    return sqrt(d);
}

// set a scalar in GPU
__global__ void set_scalar(float *p, float v){ *p = v; }

__global__ void compute_forces(float *U_out, float *F, float *x, float *m, int N, float epsilon)
{
    __shared__ float x_tile[TILE_SIZE * 3], m_tile[TILE_SIZE]; // copy onto shared memory the data required for computation (squeeze x onto 1d)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x; // indices

    // variables initialization (ease of access to particle i: each thread takes care of one particle)
    float Fi0 = 0, Fi1 = 0, Fi2 = 0, Ui = 0;
    float xi0 = (i < N) ? x[IDX(i,0)] : 0;
    float xi1 = (i < N) ? x[IDX(i,1)] : 0;
    float xi2 = (i < N) ? x[IDX(i,2)] : 0;
    float mi  = (i < N) ? m[i]        : 0;


    for(int phase = 0; phase < (N + TILE_SIZE - 1) / TILE_SIZE; phase++){ // loop across all phases (tiles)
        int j = phase * TILE_SIZE + tx;
        // load data into tile shared memory
        x_tile[tx*3] = (j<N) ? x[IDX(j,0)] : 0;
        x_tile[tx*3+1] = (j<N) ? x[IDX(j,1)] : 0;
        x_tile[tx*3+2] = (j<N) ? x[IDX(j,2)] : 0;
        m_tile[tx] = (j<N) ? m[j] : 0;

        __syncthreads(); // ensure sync of memory loads
        
        // perform computations onto the tile
        for(int k = 0; k < TILE_SIZE; k++){
            int k_absolute = phase * TILE_SIZE + k;
            // avoid: computing self-forces and non-populated memory cells
            if(k_absolute >= N || k_absolute == i) continue;
            float dx = xi0 - x_tile[k*3];
            float dy = xi1 - x_tile[k*3+1];
            float dz = xi2 - x_tile[k*3+2];
            float r2 = dx * dx + dy *dy + dz * dz + epsilon * epsilon;
            
            float r = sqrtf(r2);
            float r3 = r2 * r;
            float mimj = mi * m_tile[k]; // precomputing mimj to save 2 calc
            Fi0 -= mimj * dx / r3;
            Fi1 -= mimj * dy / r3;
            Fi2 -= mimj * dz / r3;
            if (k_absolute < i) Ui -= mimj / r; 
        }
        __syncthreads(); // sync before writing shared memory
    }
    if (i < N){
        F[IDX(i,0)] = Fi0; // store in shared mem forces
        F[IDX(i,1)] = Fi1;
        F[IDX(i,2)] = Fi2;
        atomicAdd(U_out, Ui); // update U by adding    
    }
}

__global__ void compute_accelerations(int N, float *F, float *m, float *a)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    a[IDX(i,0)] = F[IDX(i,0)] / m[i];
    a[IDX(i,1)] = F[IDX(i,1)] / m[i];
    a[IDX(i,2)] = F[IDX(i,2)] / m[i];
}

__global__ void euler_update(int N, float *x, float *v, float *a, float dt){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return; 
    v[IDX(i,0)] += a[IDX(i,0)] * dt;
    v[IDX(i,1)] += a[IDX(i,1)] * dt;
    v[IDX(i,2)] += a[IDX(i,2)] * dt;

    x[IDX(i,0)] += v[IDX(i,0)] * dt;
    x[IDX(i,1)] += v[IDX(i,1)] * dt;
    x[IDX(i,2)] += v[IDX(i,2)] * dt;
    
}

__global__ void kinetic_energy(int N, float *v, float *m, float *K_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float v2 = v[IDX(i,0)]*v[IDX(i,0)]
              + v[IDX(i,1)]*v[IDX(i,1)]
              + v[IDX(i,2)]*v[IDX(i,2)];
    atomicAdd(K_out, 0.5f * m[i] * v2);
}

__global__ void verlet_update_x(int N, float *x, float *v, float *a, float dt, float dt2){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return; 

    x[IDX(i,0)] += v[IDX(i,0)] * dt + 0.5 * a[IDX(i,0)] * dt2;
    x[IDX(i,1)] += v[IDX(i,1)] * dt + 0.5 * a[IDX(i,1)] * dt2;
    x[IDX(i,2)] += v[IDX(i,2)] * dt + 0.5 * a[IDX(i,2)] * dt2;
    
}

__global__ void verlet_update_v(int N, float *x, float *v, float *a, float *old_a, float dt){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return; 

    v[IDX(i,0)] += 0.5 * (a[IDX(i,0)] + old_a[IDX(i,0)]) * dt;
    v[IDX(i,1)] += 0.5 * (a[IDX(i,1)] + old_a[IDX(i,1)]) * dt;
    v[IDX(i,2)] += 0.5 * (a[IDX(i,2)] + old_a[IDX(i,2)]) * dt;
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

__global__ void max_acceleration(int N, float *a, float *maxa2_out)
{
    __shared__ float a_tile[TILE_SIZE]; 
    int i  = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    
    // each thread computes module of accelerations in a tile
    float a2 = 0;
    if(i<N){
        a2 = a[IDX(i,0)] * a[IDX(i,0)] + a[IDX(i,1)] * a[IDX(i,1)] + a[IDX(i,2)] * a[IDX(i,2)];
    }
    a_tile[tx] = a2;
    __syncthreads(); // wait for all threads to compute modules
    // reduce block in half (tree reduction)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tx < stride)
            a_tile[tx] = fmaxf(a_tile[tx], a_tile[tx + stride]);
        __syncthreads();
    }

    // thread 0 of each block writes the block's max to global memory
    if (tx == 0)
        atomicMax_float(maxa2_out, a_tile[0]);
    

}  


__global__ void min_distance(int N, float *x, float *mind_out)
{
    int i  = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;

    __shared__ float sx[TILE_SIZE * 3];  // tile of j-positions
    __shared__ float s[TILE_SIZE];       // reduction buffer

    float xi0 = (i < N) ? x[IDX(i,0)] : 0;
    float xi1 = (i < N) ? x[IDX(i,1)] : 0;
    float xi2 = (i < N) ? x[IDX(i,2)] : 0;

    float local_min = FLT_MAX;  // each thread tracks its own minimum

    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < num_tiles; tile++) {
        // load tile of j-particles into shared memory
        int j = tile * TILE_SIZE + tx;
        sx[tx*3+0] = (j < N) ? x[IDX(j,0)] : FLT_MAX;
        sx[tx*3+1] = (j < N) ? x[IDX(j,1)] : 0;
        sx[tx*3+2] = (j < N) ? x[IDX(j,2)] : 0;
        __syncthreads();

        // compare particle i against all j in this tile
        if (i < N) {
            for (int jj = 0; jj < TILE_SIZE; jj++) {
                int j_global = tile * TILE_SIZE + jj;
                if (j_global >= N || j_global == i) continue;
                float dx = xi0 - sx[jj*3+0];
                float dy = xi1 - sx[jj*3+1];
                float dz = xi2 - sx[jj*3+2];
                float d2 = dx*dx + dy*dy + dz*dz;
                if (d2 < local_min) local_min = d2;  // compare squared — no sqrtf yet
            }
        }
        __syncthreads();
    }

    // ── reduction: find min across all threads in block ──────────────────
    s[tx] = local_min;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tx < stride)
            s[tx] = fminf(s[tx], s[tx + stride]);
        __syncthreads();
    }

    // thread 0 writes block minimum to global result
    if (tx == 0)
        atomicMin_float(mind_out, s[0]);
}


int main(int argc, char **argv) {
    srand(time(NULL));

    int update = 1; // 0 Euler -- 1 Verlet
    int N = 1000000;  // Num particles
    float L = 1000;  // box dim
    float min_dist = 10;  // minimum distance
    bool ensurement = false; //flag
    float dt = 0.001; // time unit
    float dt2 = dt * dt; //computing dt^2 out of the loop saves time
    int iterations = 10; // tot iter
    float epsilon = 0.2; // softening constant to avoid div by zero 
    float zero = 0.0f;
    float big  = FLT_MAX;
   
    
    
    // CUDA setup
    // required memory
    size_t bytes_x = N * 3 * sizeof(float);  // memory space needed to store x
    size_t bytes_m = N * sizeof(float);      // memory space needed to store m
    size_t bytes_v = N * 3 * sizeof(float);  // memory space needed to store v
    size_t bytes_a = N * 3 * sizeof(float);  // memory space needed to store a
    size_t bytes_F = N * 3 * sizeof(float);// memory space needed to store overall F onto partile i
    size_t bytes_U = sizeof(float); 

    // varibales allocation (SHARED)
    float *x, *v, *a, *old_a, *m, *F, *U, *maxa2, *mind, *K;
    CUDA_CHECK(cudaMallocManaged(&x,     bytes_x));
    CUDA_CHECK(cudaMallocManaged(&v,     bytes_v));
    CUDA_CHECK(cudaMallocManaged(&a,     bytes_a));
    CUDA_CHECK(cudaMallocManaged(&old_a, bytes_a));
    CUDA_CHECK(cudaMallocManaged(&m,     bytes_m));
    CUDA_CHECK(cudaMallocManaged(&F,     bytes_F));
    CUDA_CHECK(cudaMallocManaged(&U,     bytes_U));
    CUDA_CHECK(cudaMallocManaged(&maxa2, sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&mind,  sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&K,     sizeof(float)));

    // before initialization ensure sync
    cudaDeviceSynchronize();

    // initial conditions (sequential)
    for (int i = 0; i < N; i++)
        m[i] = ((float)rand() / (float)(RAND_MAX) * 9) + 1;

    
    int it = 0;
    while (it < N) {
        for (int j = 0; j < 3; j++)
            x[IDX(it,j)] = (float)rand() / (float)RAND_MAX * L;

        ensurement = true;
        /*
        for (int k = 0; k < it; k++) {
            if (distance(&x[IDX(it,0)], &x[IDX(k,0)]) < min_dist) {
                ensurement = false;
                break;
            }
        */
        if (ensurement) {
            //printf("Placed particle %d\n", it);
            it++;
        }
    }
    
    for (int i = 0; i < N; i++)
        for (int j = 0; j < 3; j++)
            v[IDX(i,j)] = (float)rand() / (float)RAND_MAX * 2 - 1;  // v in [-1, +1]

    // print initial conditions
    printf("initial positions (first 100):\n");
    for (int i = 0; i < 5; i++) {
        printf("particle %d: ", i);
        for (int j = 0; j < 3; j++)
            printf("%.6f ", x[IDX(i,j)]);
        printf("\n");
    }
    printf("masses:\n");
    for (int i = 0; i < 5; i++)
        printf("particle %d: %f\n", i, m[i]);

    printf("initial velocities:\n");
    for (int i = 0; i < 5; i++) {
        printf("particle %d: ", i);
        for (int j = 0; j < 3; j++)
            printf("%.6f ", v[IDX(i,j)]);
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


    // copying values to CUDA (GPU to global memory)
    // NOT NEEDED WITH SHARED MEMORY
    /*cudaMemcpy(d_x, h_x, bytes_x, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, bytes_v, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, h_m, bytes_m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, h_F, bytes_F, cudaMemcpyHostToDevice);
    cudaMemcpy(d_U, h_U, bytes_U, cudaMemcpyHostToDevice);
    cudaMemcpy(d_old_a, h_old_a, bytes_a, cudaMemcpyHostToDevice);
    */
    
  
        
    // CUDA dimensions
    int THREADS = TILE_SIZE;
    int BLOCKS = (N + THREADS - 1) / THREADS;  // add this
    dim3 threadsPerBlock(THREADS), numBlocks(BLOCKS);

    // CUDA timing
    cudaEvent_t ev_start, ev_stop;

    // iteration loop EULER
    if (update == 0){
    for (int t = 0; t < iterations; t++) {
        int diag = (t % 100 == 0);
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_stop);
        cudaEventRecord(ev_start);

        // forces kernel
        *U = 0.0f;
        *K = 0.0f;
        compute_forces<<<numBlocks, threadsPerBlock>>>(U, F, x, m, N, epsilon);
        // acceleration kernel
        compute_accelerations<<<numBlocks, threadsPerBlock>>>(N, F, m, a);
        
        kinetic_energy<<<numBlocks, threadsPerBlock>>>(N, v, m, K);
        euler_update<<<numBlocks, threadsPerBlock>>>(N, x, v, a, dt);
        kernel_boundary_conditions<<<numBlocks, threadsPerBlock>>>(N, x, L);
        // stop timer after integration
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop); // wait for GPU to finish
        float ms;
        cudaEventElapsedTime(&ms, ev_start, ev_stop); // computation time
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_stop);
        
        // diagnosis
        if (diag){
            // compute max accel
            *maxa2 = 0.0f; 
            *mind = FLT_MAX;
            max_acceleration<<<BLOCKS, THREADS>>>(N, a, maxa2);  // look for highest modulo
            
            // compute min distance
            min_distance<<<BLOCKS, THREADS>>>(N, x, mind); // look for minimum distance
            cudaDeviceSynchronize();
            float maxa = sqrtf(*maxa2);
            float mind_val = sqrtf(*mind);
            float mint = sqrtf(mind_val / maxa);
            printf("minimum dt: %.10f\n",mint);

        }
        
        if (t % 100 == 0){
                printf("%d\n",t);
            }
        //write to CSV time information
        fprintf(tp, "%d,%.10f\n", t, ms/1000.0f);
        //write to CSV energy information
        fprintf(ep, "%d,%.6f,%.6f,%.6f\n", t, *U, *K, *U + *K);
        
    }
}
    // iteration loop VERLET
    if (update == 1){
        cudaMemsetAsync(U,0,sizeof(float));
        // first verlet iter
        compute_forces<<<numBlocks, threadsPerBlock>>>(U, F, x, m, N, epsilon);
        compute_accelerations<<<numBlocks, threadsPerBlock>>>(N, F, m, a);

        for (int t = 0; t < iterations; t++) {
            int diag = (t % 100 == 0);
            cudaEventCreate(&ev_start);
            cudaEventCreate(&ev_stop);
            cudaEventRecord(ev_start);

            // step 1
            verlet_update_x<<<numBlocks, threadsPerBlock>>>(N, x, v, a, dt, dt2);
            kernel_boundary_conditions<<<numBlocks, threadsPerBlock>>>(N, x, L);

            // step 2    
            cudaMemcpyAsync(old_a, a, bytes_a, cudaMemcpyDeviceToDevice);

            cudaMemsetAsync(U,0,sizeof(float)); // reset U as zero 
            compute_forces<<<numBlocks, threadsPerBlock>>>(U, F, x, m, N, epsilon);
            compute_accelerations<<<numBlocks, threadsPerBlock>>>(N, F, m, a);
            // step 3
            verlet_update_v<<<numBlocks, threadsPerBlock>>>(N, x, v, a, old_a, dt);
            // kinetic energy kernel
            cudaMemsetAsync(K,0,sizeof(float)); // reset K as zero
            kinetic_energy<<<numBlocks, threadsPerBlock>>>(N, v, m, K);
            
            // stop timer after integration
            cudaEventRecord(ev_stop);
            cudaEventSynchronize(ev_stop); // wait for GPU to finish
            float ms;
            cudaEventElapsedTime(&ms, ev_start, ev_stop); // computation time
            cudaEventDestroy(ev_start);
            cudaEventDestroy(ev_stop);
            

            // diagnosis
            if (diag){
                // compute max accel
                cudaMemsetAsync(maxa2,0,sizeof(float)); // reset a2 as zero 
                set_scalar<<<1,1>>>(mind,FLT_MAX); // set mind dist as max
                max_acceleration<<<BLOCKS, THREADS>>>(N, a, maxa2);  // look for highest modulo
                
                // compute min distance
                min_distance<<<BLOCKS, THREADS>>>(N, x, mind); // look for minimum distance
                cudaDeviceSynchronize();
                float maxa = sqrtf(*maxa2);
                float mind_val = sqrtf(*mind);
                float mint = (maxa > 0.0f) ? sqrtf(mind_val / maxa) : 0.0f;
                printf("minimum dt: %.10f\n",mint);
            }
            
            cudaDeviceSynchronize();
            //write to CSV time information
            fprintf(tp, "%d,%.10f\n", t, ms/1000.0f);
            //write to CSV energy information
            fprintf(ep, "%d,%.6f,%.6f,%.6f\n", t, *U, *K, *U + *K);
            if (t % 100 == 0){
                printf("%d\n",t);
            }
        }
    }
        
        

        
       
    


    

    fclose(fp);
    fclose(tp);   // add
    fclose(ep);   // add
    return 0;
}