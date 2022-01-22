//
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


//
typedef float              f32;
typedef double             f64;
typedef unsigned long long u64;


typedef struct{
  f32 *x, *y, *z;
} vector; 

typedef struct particle_s {

  vector positions, velocities;
  
} particle_t;


void init(particle_t *p, u64 n)
{

  p->positions.x = calloc(n, n * sizeof(f32));
  p->positions.y = calloc(n, n * sizeof(f32));
  p->positions.z = calloc(n, n * sizeof(f32));

  p->velocities.x = calloc(n, n * sizeof(f32));
  p->velocities.y = calloc(n, n * sizeof(f32));
  p->velocities.z = calloc(n, n * sizeof(f32));

  u64 r1 = (u64)rand();
  u64 r2 = (u64)rand();
  f32 sign = (r1 > r2) ? 1 : -1;

  for (u64 i = 0; i < n; i++)
    {  
      //
      p->positions.x[i] = sign * (f32)rand() / RAND_MAX;
      p->positions.y[i] = (f32)rand() / RAND_MAX;
      p->positions.z[i] = sign * (f32)rand() / RAND_MAX;

      //
      p->velocities.x[i] = (f32)rand() / RAND_MAX;
      p->velocities.y[i] = sign * (f32)rand() / RAND_MAX;
      p->velocities.z[i] = (f32)rand() / RAND_MAX;
    }
}

void move_particles(particle_t *p, const f32 dt, u64 n)
{
  //
  const f32 softening = 1e-20;

  //
  vector f; 
  f.x = calloc(n,sizeof(f32));
  f.y = calloc(n,sizeof(f32));
  f.z = calloc(n,sizeof(f32));

  vector d;
  d.x = calloc(n,sizeof(f32));
  d.y = calloc(n,sizeof(f32));
  d.z = calloc(n,sizeof(f32));


  //
  for (u64 i = 0; i < n; i++){
    
      //23 floating-point operations
      for (u64 j = 0; j < n; j++)
	    {
	  //Newton's law
      
        d.x[i] = p->positions.x[j] - p->positions.x[i]; //1
        d.y[i] = p->positions.y[j] - p->positions.y[i]; //2 
        d.z[i] = p->positions.z[j] - p->positions.z[i]; //3
      
      
	    const f32 d_2 = (d.x[i] * d.x[i]) + (d.y[i] * d.y[i]) + (d.z[i] * d.z[i]) + softening; //9
	    const f32 d_3_over_2 = pow(d_2, 3.0 / 2.0); //11
      

	  //Net force
	    f.x[i] += d.x[i] / d_3_over_2; //13
	    f.y[i] += d.y[i] / d_3_over_2; //15
	    f.z[i] += d.z[i] / d_3_over_2; //17
	}

      //
      p->velocities.x[i] += dt * f.x[i];//19
      p->velocities.y[i] += dt * f.y[i];//21
      p->velocities.z[i] += dt * f.z[i];//23
    }

  //3 floating-point operations
  for (u64 i = 0; i < n; i++)
    {
      p->positions.x[i] += dt * p->velocities.x[i];
      p->positions.y[i] += dt * p->velocities.y[i];
      p->positions.z[i] += dt * p->velocities.z[i];
    }
}

//
int main(int argc, char **argv)
{
  //
  const u64 n = (argc > 1) ? atoll(argv[1]) : 16384;
  const u64 steps= 10;
  const f32 dt = 0.01;

  //
  f64 rate = 0.0, drate = 0.0;

  //Steps to skip for warm up
  const u64 warmup = 3;
  
  //
  particle_t *p = calloc(n,sizeof(particle_t) * n);

  //
  init(p, n);

  const u64 s = sizeof(particle_t) * n;
  
  printf("\n\033[1mTotal memory size:\033[0m %llu B, %llu KiB, %llu MiB\n\n", s, s >> 10, s >> 20);
  
  //
  printf("\033[1m%5s %10s %10s %8s\033[0m\n", "Step", "Time, s", "Interact/s", "GFLOP/s"); fflush(stdout);
  
  //
  for (u64 i = 0; i < steps; i++)
    {
      //Measure
      const f64 start = omp_get_wtime();

      move_particles(p, dt, n);

      const f64 end = omp_get_wtime();

      //Number of interactions/iterations
      const f32 h1 = (f32)(n) * (f32)(n - 1);

      //GFLOPS
      const f32 h2 = (23.0 * h1 + 3.0 * (f32)n) * 1e-9;
      
      if (i >= warmup)
	{
	  rate += h2 / (end - start);
	  drate += (h2 * h2) / ((end - start) * (end - start));
	}

      //
      printf("%5llu %10.3e %10.3e %8.1f %s\n",
	     i,
	     (end - start),
	     h1 / (end - start),
	     h2 / (end - start),
	     (i < warmup) ? "*" : "");
      
      fflush(stdout);
    }

  //
  rate /= (f64)(steps - warmup);
  drate = sqrt(drate / (f64)(steps - warmup) - (rate * rate));

  printf("-----------------------------------------------------\n");
  printf("\033[1m%s %4s \033[42m%10.1lf +- %.1lf GFLOP/s\033[0m\n",
	 "Average performance:", "", rate, drate);
  printf("-----------------------------------------------------\n");
  
  //
  free(p);


  //
  return 0;
}
