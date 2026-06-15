#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <float.h>

#define THREADS_PER_BLOCK 128
#define IDX(i,k) ((i)*3+(k))
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d — %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)


// set a scalar in GPU
__global__ void set_scalar(float *p, float v){ *p = v; }

__global__ void kernel_forces(int N, float epsilon, float epsilon2, float *x, float *m, float *F, float *U_out)
{
    __shared__ float x_tile[THREADS_PER_BLOCK * 3], m_tile[THREADS_PER_BLOCK]; // copy onto shared memory the data required for computation 
    // this setup allows for launching in chunks (avoid timeout)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return; // memory bound check
    int tx = threadIdx.x; // indices

    // variables initialization (ease of access to particle i: each thread takes care of one particle)
    float Fi0 = 0, Fi1 = 0, Fi2 = 0, Ui = 0;
    float xi0 = (i < N) ? x[IDX(i,0)] : 0;
    float xi1 = (i < N) ? x[IDX(i,1)] : 0;
    float xi2 = (i < N) ? x[IDX(i,2)] : 0;
    float mi  = (i < N) ? m[i]        : 0;

    for(int phase = 0; phase < (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; phase++){ // loop across all phases (tiles)
        int j = phase * THREADS_PER_BLOCK + tx;
        // load data into tile shared memory
        x_tile[tx*3] = (j<N) ? x[IDX(j,0)] : 0;
        x_tile[tx*3+1] = (j<N) ? x[IDX(j,1)] : 0;
        x_tile[tx*3+2] = (j<N) ? x[IDX(j,2)] : 0;
        m_tile[tx] = (j<N) ? m[j] : 0;

        __syncthreads(); // ensure sync of memory loads
        
        // perform computations onto the tile
        for(int k = 0; k < THREADS_PER_BLOCK; k++){
            int k_absolute = phase * THREADS_PER_BLOCK + k;
            // avoid: computing self-forces and non-populated memory cells
            if(k_absolute >= N || k_absolute == i) continue;
            float dx = xi0 - x_tile[k*3];
            float dy = xi1 - x_tile[k*3+1];
            float dz = xi2 - x_tile[k*3+2];
            float r2 = dx * dx + dy *dy + dz * dz + epsilon2;
            
            float inv_r = rsqrtf(r2);
            float inv_r3 = inv_r * inv_r * inv_r;
            float mimj = mi * m_tile[k]; // precomputing mimj to save 2 calc
            Fi0 -= mimj * dx * inv_r3;
            Fi1 -= mimj * dy * inv_r3;
            Fi2 -= mimj * dz * inv_r3;
            if (k_absolute < i) Ui -= mimj * inv_r; 
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

// for initialization
float gaussian(){
    float u1 = (rand()+1.0) / ((float)RAND_MAX + 2.0); // U(0,1)
    float u2 = (rand()+1.0) / ((float)RAND_MAX + 2.0); // U(0,1)
    return sqrt(-2.0 * log(u1)) * cos(6.2831855307 * u2); 
}


int main(int argc, char **argv) {
    srand(3110);

    int update = 1; // 0 Euler -- 1 Verlet
    int star = 0;   // 0: no star at middle -- 1 star at middle
    int mb = 1;     // 0: random initialization of v[-1,1] -- 1: Maxwell-Boltzmann like distribution

    int iterations = 10000; // tot iter
    int N = 50000; // Num particles
    float T = 2.0; // for maxwell boltzmann distr.
    int custom_ic = 0; // custom initial cond. for plotting trajectories

    float L = 1000;  // box dim
    float min_dist = 10;  // minimum distance
    bool ensurement = false; //flag
    float dt = 0.04; // time unit
    float dt2 = dt * dt; //computing dt^2 out of the loop saves time
    clock_t start, end; //time measurement
    float diff;        //time measurement
    float epsilon = 0.2; // softening constant to avoid div by zero
    float big = FLT_MAX;
    float epsilon2 = epsilon * epsilon;
   
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

    /// initialization (CPU)
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
            x[IDX(it,j)] = (float)rand() / (float)RAND_MAX * L;

        // first particle is the star
        if (star == 1 && it == 0){
            for (int j = 0; j < 3; j++){
                x[IDX(0,j)] = (float)L/2.0;
            }
        }
        ensurement = true;
        if (N < 12000){
            for (int j = 0; j < it; j++) {
                float dx = x[IDX(it,0)]-x[IDX(j,0)], dy = x[IDX(it,1)]-x[IDX(j,1)], dz = x[IDX(it,2)]-x[IDX(j,2)];
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
                v[IDX(i,j)] = (float)rand() / (float)RAND_MAX * 2 - 1;  // v in [-1, +1]

        if (star == 1){
            for (int j = 0; j < 3; j++)
                v[IDX(0,j)] = 0.0;  // star does not move at beginning
        }
    }
    // MAXWELL-BOLTZMANN branch
    if(mb == 1){
        for(int i = 0; i < N; i++){
            float sigma = sqrt(T / m[i]);  // v ~ N(0,kb*T/M)
            for(int j = 0; j<3; j++){
                v[IDX(i,j)] = sigma * gaussian(); 
            }
        }
    }

    if(custom_ic == 1){
        // masses
        m[0] = 10.0;
        m[1] = 10.0;

        x[IDX(0,0)] = 400; x[IDX(0,1)] = 550; x[IDX(0,2)] = 550;
        x[IDX(1,0)] = 600; x[IDX(1,1)] = 500; x[IDX(1,2)] = 550;

        v[IDX(0,0)] = 0.0; v[IDX(0,1)] =  0.15; v[IDX(0,2)] = 0.0;
        v[IDX(1,0)] = 0.0; v[IDX(1,1)] = -0.15; v[IDX(1,2)] = 0.0;
    }

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
    FILE *vp = fopen("velocity.csv", "w");
    if (vp == NULL) {
        fprintf(stderr, "Error opening file\n");
        return 1;
    }
    fprintf(vp, "timestep,particle,vx,vy,vz\n");
    fprintf(fp, "timestep,particle,x,y,z\n");
    fprintf(tp, "timestep,runtime,kernelruntime\n");
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
    int THREADS = THREADS_PER_BLOCK;
    int BLOCKS = (N + THREADS - 1) / THREADS;  // add this
    dim3 threadsPerBlock(THREADS), numBlocks(BLOCKS);

    // CUDA timing
    cudaEvent_t ev_start, ev_stop;

    // iteration loop EULER
    if (update == 0){
    for (int t = 0; t < iterations; t++) {
        int diag = (t % 100 == 0 && t > 0);

        *U = 0.0f;
        *K = 0.0f;
        cudaEvent_t ev_start, ev_stop, kernel_start, kernel_stop;
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_stop);
        cudaEventCreate(&kernel_start);
        cudaEventCreate(&kernel_stop);
        cudaEventRecord(ev_start);

        // forces kernel
        cudaEventRecord(kernel_start);
        kernel_forces<<<BLOCKS,THREADS>>>(N,epsilon,epsilon2, x, m, F, U);
        CUDA_CHECK(cudaGetLastError());
        cudaEventRecord(kernel_stop);
        cudaDeviceSynchronize();
        cudaEventSynchronize(kernel_stop);
        float kms;
        cudaEventElapsedTime(&kms, kernel_start, kernel_stop);
        cudaEventDestroy(kernel_start);
        cudaEventDestroy(kernel_stop);
        // acceleration kernel
        compute_accelerations<<<numBlocks, threadsPerBlock>>>(N, F, m, a);
        
        kinetic_energy<<<numBlocks, threadsPerBlock>>>(N, v, m, K);
        euler_update<<<numBlocks, threadsPerBlock>>>(N, x, v, a, dt);
        //kernel_boundary_conditions<<<numBlocks, threadsPerBlock>>>(N, x, L);
        // stop timer after integration
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop); // wait for GPU to finish
        float ms;
        cudaEventElapsedTime(&ms, ev_start, ev_stop); // computation time
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_stop);
       
        if (t % 100 == 0){
                printf("%d\n",t);
            }
        //write to CSV time information
        fprintf(tp, "%d,%.10f,%.10f\n", t, ms/1000.0f, kms/1000.0f);
        //write to CSV energy information
        fprintf(ep, "%d,%.6f,%.6f,%.6f\n", t, *U, *K, *U + *K);
        for (int i = 0; i < N; i++)
                fprintf(fp, "%d,%d,%.6f,%.6f,%.6f\n", t, i, x[IDX(i,0)], x[IDX(i,1)], x[IDX(i,2)]);
            printf("%.6f,%.6f,%.6f\n",x[IDX(0,0)], x[IDX(0,1)], x[IDX(0,2)]);
            for (int i = 0; i < N; i++)
                fprintf(vp, "%d,%d,%.6f,%.6f,%.6f\n", t, i, v[IDX(i,0)], v[IDX(i,1)], v[IDX(i,2)]);
            if (t % 100 == 0){
                printf("%d\n",t);
            }
        
    }
}
    // iteration loop VERLET
    if (update == 1){
        cudaMemsetAsync(U,0,sizeof(float));
        // first verlet iter
        kernel_forces<<<BLOCKS,THREADS>>>(N,epsilon,epsilon2, x, m, F, U);
        compute_accelerations<<<numBlocks, threadsPerBlock>>>(N, F, m, a);

        for (int t = 0; t < iterations; t++) {
            int diag = (t % 100 == 0 && t > 0);
            cudaEventCreate(&ev_start);
            cudaEventCreate(&ev_stop);
            cudaEventRecord(ev_start);

            // step 1
            verlet_update_x<<<numBlocks, threadsPerBlock>>>(N, x, v, a, dt, dt2);
            //kernel_boundary_conditions<<<numBlocks, threadsPerBlock>>>(N, x, L);

            // step 2    
            cudaMemcpyAsync(old_a, a, bytes_a, cudaMemcpyDeviceToDevice);

            cudaMemsetAsync(U,0,sizeof(float)); // reset U as zero 
            kernel_forces<<<BLOCKS,THREADS>>>(N,epsilon,epsilon2, x, m, F, U);
            
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
                        
            cudaDeviceSynchronize();
            if (t % 20 == 0){
                //write to CSV time information
                fprintf(tp, "%d,%.10f\n", t, ms/1000.0f);
                //write to CSV energy information
                fprintf(ep, "%d,%.6f,%.6f,%.6f\n", t, *U, *K, *U + *K);
                // write to CSV
                for (int i = 0; i < N; i++)
                    fprintf(fp, "%d,%d,%.6f,%.6f,%.6f\n", t, i, x[IDX(i,0)], x[IDX(i,1)], x[IDX(i,2)]);
                printf("%.6f,%.6f,%.6f\n",x[IDX(0,0)], x[IDX(0,1)], x[IDX(0,2)]);
                for (int i = 0; i < N; i++)
                    fprintf(vp, "%d,%d,%.6f,%.6f,%.6f\n", t, i, v[IDX(i,0)], v[IDX(i,1)], v[IDX(i,2)]);
                
                printf("%d\n",t);
            }
        }
    }
        
        

        
       
    


    

    fclose(fp);
    fclose(tp);   // add
    fclose(ep);   // add
    fclose(vp);   // add
    return 0;
}
