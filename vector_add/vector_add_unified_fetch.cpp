// This program computer the sum of two N-element vectors using unified memory
// By: Nick from CoffeeBeforeArch

#include <stdio.h>
#include <cassert>
#include <iostream>

using std::cout;

// CUDA kernel for vector addition
// No change when using CUDA unified memory
__global__ void vectorAdd(int *a, int *b, int *c, int N) {
  // Calculate global thread thread ID
  int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

  // Boundary check
  if (tid < N) {
    c[tid] = a[tid] + b[tid];
  }
}

int main() {
  // Array size of 2^16 (65536 elements)
  const int N = 1 << 16;
  size_t bytes = N * sizeof(int);

  // Declare unified memory pointers
  int *a, *b, *c;

  // Allocation memory for these pointers
  cudaMallocManaged(&a, bytes);
  cudaMallocManaged(&b, bytes);
  cudaMallocManaged(&c, bytes);
  
  // Get the device ID for prefetching calls
  int id = cudaGetDevice(&id);

  // Set some hints about the data and do some prefetching
  cudaMemAdvise(a, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
  cudaMemAdvise(b, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
  cudaMemPrefetchAsync(c, bytes, id);

  // Initialize vectors
  for (int i = 0; i < N; i++) {
    a[i] = rand() % 100;
    b[i] = rand() % 100;
  }
  
  // Pre-fetch 'a' and 'b' arrays to the specified device (GPU)
  cudaMemAdvise(a, bytes, cudaMemAdviseSetReadMostly, id);
  cudaMemAdvise(b, bytes, cudaMemAdviseSetReadMostly, id);
  cudaMemPrefetchAsync(a, bytes, id);
  cudaMemPrefetchAsync(b, bytes, id);
  
  // Threads per CTA (1024 threads per CTA)
  int BLOCK_SIZE = 1 << 10;

  // CTAs per Grid
  int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // Call CUDA kernel
  vectorAdd<<<GRID_SIZE, BLOCK_SIZE>>>(a, b, c, N);

  // Wait for all previous operations before using values
  // We need this because we don't get the implicit synchronization of
  // cudaMemcpy like in the original example
  cudaDeviceSynchronize();

  // Prefetch to the host (CPU)
  cudaMemPrefetchAsync(a, bytes, cudaCpuDeviceId);
  cudaMemPrefetchAsync(b, bytes, cudaCpuDeviceId);
  cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);

  // Verify the result on the CPU
  for (int i = 0; i < N; i++) {
    assert(c[i] == a[i] + b[i]);
  }

  // Free unified memory (same as memory allocated with cudaMalloc)
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);

  cout << "COMPLETED SUCCESSFULLY!\n";

  return 0;
}


/*

1. `int *a, *b, *c;`: Bu satır, üç tane `int` işaretçisi tanımlar. Bu işaretçiler, 
GPU belleğine (`cudaMallocManaged` kullanılarak ayrılmış) veya CPU belleğine işaret edebilirler.

2. `cudaMallocManaged(&a, bytes);`, `cudaMallocManaged(&b, bytes);`, 
ve `cudaMallocManaged(&c, bytes);`: Bu satırlar, her bir işaretçiye bellek tahsis eder. `cudaMallocManaged`, 
hem CPU hem de GPU tarafından erişilebilen birleştirilmiş bir bellek alanı (Unified Memory) tahsis eder.
 Bu, veriyi CPU ve GPU arasında otomatik olarak taşımanıza ve erişmenize olanak tanır.

3. `int id = cudaGetDevice(&id);`: Bu satırda, `cudaGetDevice` işlevi ile mevcut CUDA cihazının kimliği (`id`) alınır. 
Ancak, `cudaGetDevice` işlevi ile döndürülen değer, `id` adlı değişkene yanlışlıkla atanmış gibi görünüyor. 
Doğru bir kullanım, `int id; cudaGetDevice(&id);` olmalıdır.

4. `cudaMemAdvise(a, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);` ve 
`cudaMemAdvise(b, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);`: 
Bu satırlar, `a` ve `b` işaretçilerinin CPU tarafından tercih edilen bir konumda olmasını belirtir. 
Yani, bu verilerin CPU belleğinde bulunmasını önerir.

5. `cudaMemPrefetchAsync(c, bytes, id);`: Bu satır, `c` işaretçisini GPU'ya önceden yükler (prefetch) 
ve belirtilen CUDA cihazı kimliği (`id`) ile ilişkilendirir. Bu, `c`'nin GPU belleğine taşınmasını ve 
GPU tarafından daha hızlı erişilmesini sağlar.

Sonuç olarak, `c` işaretçisinin tanımı ve veri yerleştirme (placement) önerileri ile `a` ve `b` işaretçilerinin 
tanımı arasındaki fark, `c`'nin GPU tarafından önceden yüklenmesi ve diğer iki işaretçinin CPU tarafından 
tercih edilen bir konumda olmasıdır. Bu tür bellek yönetimi ve yerleştirme stratejileri, veri erişimini ve 
performansı optimize etmek için kullanılır.

*/