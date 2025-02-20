#include <omp.h>
#include <stdio.h>
#include <time.h>

#define N 100000000

int main() {
    double start, end, t;
    long long seq=0, para=0;

    start = omp_get_wtime();
    for (int i = 0; i < N; i++) {
        seq += i;
    }
    end = omp_get_wtime();
    t = end - start;
    printf("Sequential Execution time: %f seconds\n", t);

    // Must assign: ordered in/outside for loop
    start = omp_get_wtime();
    #pragma omp parallel for reduction(+: para) ordered
    for (int i = 0; i < N; i++) {
        #pragma omp ordered
        para += i;
    }
    end = omp_get_wtime();
    t = end - start;
    printf("Execution time: %f seconds\n", t);



    #pragma omp parallel sections
    {
        printf("#thrads: %d\n", omp_get_num_threads());

        #pragma omp section
        {
            printf("Section 1: %d\n", omp_get_thread_num());
            // Code for section 1
        }

        #pragma omp section
        {
            printf("Section 2: %d\n", omp_get_thread_num());
            // Code for section 2
        }

        #pragma omp section
        {
            printf("Section 3: %d\n", omp_get_thread_num());
            // Code for section 3
        }
    }
     
    system("PAUSE");
    return 0;
}

/* Compile
    gcc -fopenmp <file>.c -o <file>.exe 
*/