#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

/* Handler for mpi functions */
#define handle_mpi(func, s)      \
  if (func != MPI_SUCCESS) {     \
    if (!rank) fputs(s, stderr); \
    MPI_Finalize();              \
    exit(1);                     \
  }

/* Handler for functions with the defined error code */
#define handle_yes(func, val, s) \
  if (func == val) {             \
    if (!rank) fputs(s, stderr); \
    MPI_Finalize();              \
    exit(1);                     \
  }

/* Handler for functions without the defined error code */
#define handle_no(func, val, s)  \
  if (func != val) {             \
    if (!rank) fputs(s, stderr); \
    MPI_Finalize();              \
    exit(1);                     \
  }

/* Predefined constants */
enum NUM { ARG_NUM = 4, DIM_NUM = 3 };
enum mode { A, B };
enum FILES { EXECUTED_FILE, MATRIX_A_FILE, MATRIX_B_FILE, OUTPUT_FILE };
enum COORDS { X, Y, Z };

typedef int32_t mtype;

/* Varialbes used by all processes, that would changed only once */
static int proc_on_dim;
static mtype n;
static int block_size;
static int rank;
static int coords[DIM_NUM];

mtype *read_block(MPI_File matrix, int i, int j) {
  mtype *block = malloc(sizeof(mtype) * block_size * block_size);
  if (!block) {
    if (!rank) {
      fputs("Can't allocate the necessary memory!", stderr);
    }
    MPI_Finalize();
    exit(1);
  }

  MPI_Datatype block_type;
  handle_mpi(
      MPI_Type_vector(block_size, block_size, n, MPI_INT32_T, &block_type),
      "Can't create a new type!");
  handle_mpi(MPI_Type_commit(&block_type), "Can't commit the new type!");

  handle_mpi(
      MPI_File_set_view(matrix, ((i * n + j) * block_size + 1) * sizeof(mtype),
                        MPI_INT32_T, block_type, "native", MPI_INFO_NULL),
      "Can't set view in file!");

  handle_mpi(MPI_File_read_all(matrix, block, block_size * block_size,
                               MPI_INT32_T, MPI_STATUS_IGNORE),
             "Can't read from file!");

  handle_mpi(MPI_Type_free(&block_type), "Can't delete the new type!");

  return block;
}

void load_matrix(const char *file_name, MPI_Comm cube, mtype *n, mtype **block,
                 int mode) {
  MPI_File matrix;
  handle_mpi(
      MPI_File_open(cube, file_name, MPI_MODE_RDONLY, MPI_INFO_NULL, &matrix),
      "Can't open a file!");

  handle_mpi(MPI_File_read_all(matrix, n, 1, MPI_INT32_T, MPI_STATUS_IGNORE),
             "Can't read info from the file!");

  block_size = *n / proc_on_dim;

  switch (mode) {
    case A:
      *block = read_block(matrix, coords[0], coords[2]);
      break;
    case B:
      *block = read_block(matrix, coords[2], coords[1]);
      break;
    default:;
  }
  handle_mpi(MPI_File_close(&matrix), "Can't close the file!");
}

void print_result(const char *file_name, MPI_Comm cube, mtype *buf) {
  /* Creating a new topo to write the results */
  MPI_Comm layer;
  int remain_dims[DIM_NUM] = {1, 1, 0};
  handle_mpi(MPI_Cart_sub(cube, remain_dims, &layer),
             "Can't create a new topo {layer}!");

  if (!coords[Z]) {
    if (!coords[X] && !coords[Y]) {
      FILE *out = fopen(file_name, "wb");
      if (!out) {
        if (!rank) {
          fputs("Can't open an output file!", stderr);
        }
        MPI_Finalize();
        exit(1);
      }
      if (fwrite(&n, sizeof(n), 1, out) != 1) {
        if (!rank) {
          fputs("Can't write into the output file!", stderr);
        }
        MPI_Finalize();
        exit(1);
      }
      if (fclose(out) == EOF) {
        if (!rank) {
          fputs("Can't close the output file!", stderr);
        }
        MPI_Finalize();
        exit(1);
      }
    }

    MPI_File matrix;
    handle_mpi(
        MPI_File_open(layer, file_name, MPI_MODE_WRONLY | MPI_MODE_APPEND,
                      MPI_INFO_NULL, &matrix),
        "Can't open the output file for processes in layer!");

    MPI_Datatype block_type;
    handle_mpi(
        MPI_Type_vector(block_size, block_size, n, MPI_INT32_T, &block_type),
        "Can't create a new MPI_Type for writing the results!");

    handle_mpi(MPI_Type_commit(&block_type),
               "Can't commit the new MPI_Type for writing");

    handle_mpi(
        MPI_File_set_view(
            matrix,
            ((coords[X] * n + coords[Y]) * block_size + 1) * sizeof(mtype),
            MPI_INT32_T, block_type, "native", MPI_INFO_NULL),
        "Can't set view in the output file!");

    handle_mpi(MPI_File_write(matrix, buf, block_size * block_size, MPI_INT32_T,
                              MPI_STATUS_IGNORE),
               "Can't write in the output file!");
    handle_mpi(MPI_Type_free(&block_type),
               "Can't delete the new MPI_Type for writing!");
    handle_mpi(MPI_File_close(&matrix), "Can't close the output file!");
  }
  handle_mpi(MPI_Comm_free(&layer), "Can't delete topo {layer}!");
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS) {
    MPI_Finalize();
    exit(1);
  }

  if (argc < ARG_NUM) {
    if (!rank) {
      fprintf(stderr, "usage: %s <npx> <npy> <matrix>", argv[EXECUTED_FILE]);
    }
    MPI_Finalize();
    return 1;
  }

  int np;
  handle_mpi(MPI_Comm_size(MPI_COMM_WORLD, &np),
             "Can't define the number of processes!");

  if (np == 1) {
    FILE *matrix_A = fopen(argv[MATRIX_A_FILE], "rb");
    FILE *matrix_B = fopen(argv[MATRIX_B_FILE], "rb");
    handle_yes(matrix_A, NULL, "Can't open a file with the first matrix!");
    handle_yes(matrix_B, NULL, "Can't open a file with the second matrix!");
    handle_no(fread(&n, sizeof(n), 1, matrix_A), 1,
              "Can't read info from the first file!");
    mtype *a = malloc(sizeof(*a) * n * n);
    handle_yes(a, NULL, "Can't allocate memory for the first matrix!");
    handle_no(fread(a, sizeof(*a), n * n, matrix_A), n * n,
              "Can't read the whole matrix from the first file!");
    handle_yes(fclose(matrix_A), EOF, "Can't close the first file!");

    handle_no(fseek(matrix_B, sizeof(n), SEEK_SET), 0,
              "Can't make an offset in the second file!");
    mtype *b = malloc(sizeof(*b) * n * n);
    handle_yes(b, NULL, "Can't allocate memory for the second matrix!");
    handle_no(fread(b, sizeof(*b), n * n, matrix_B), n * n,
              "Can't read the whole matrix! form the second file");
    handle_yes(fclose(matrix_B), EOF, "Can't close the second file!");

    mtype *c = calloc(n * n, sizeof(*c));
    handle_yes(c, NULL, "Can't allocate memory for the result matrix!");
    mtype temp;

    double start, end;

    start = MPI_Wtime();

    for (int i = 0; i < n; ++i) {
      for (int k = 0; k < n; ++k) {
        temp = a[i * n + k];
        for (int j = 0; j < n; ++j) {
          c[i * n + j] += temp * b[k * n + j];
        }
      }
    }

    end = MPI_Wtime();
    double time = end - start;
    double max_time;

    handle_mpi(
        MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD),
        "Can't define the elapsed time!");

    free(a);
    free(b);

    FILE *out = fopen(argv[OUTPUT_FILE], "wb");
    handle_yes(out, NULL, "Can't open file for the result matrix!");
    handle_no(fwrite(&n, sizeof(n), 1, out), 1,
              "Can't write the result into the output file!");
    handle_no(fwrite(c, sizeof(*c), n * n, out), n * n,
              "Can't write the result matrix into the output file!");
    handle_yes(fclose(out), EOF, "Can't close the output file!");

    if (!rank) {
      printf("\n\tElapsed time: %lf\n", max_time);
    }

    MPI_Finalize();
    return 0;
  }

  proc_on_dim = (int)(pow(np, 1.0 / DIM_NUM));

  int dims[DIM_NUM] = {proc_on_dim, proc_on_dim, proc_on_dim};
  int periods[DIM_NUM] = {0, 0, 0};

  /* Creating a new topo and defining coordinates of the proccess on this topo
   */
  MPI_Comm cube;
  handle_mpi(MPI_Cart_create(MPI_COMM_WORLD, DIM_NUM, dims, periods, 0, &cube),
             "Can't create a new topo!");
  handle_mpi(MPI_Cart_get(cube, DIM_NUM, dims, periods, coords),
             "Can't get coords of proc!");

  mtype *block_A = NULL, *block_B = NULL;

  /* Reading the necessary parts of matrixes */
  load_matrix(argv[MATRIX_A_FILE], cube, &n, &block_A, A);
  load_matrix(argv[MATRIX_B_FILE], cube, &n, &block_B, B);

  /* Multiplying blocks of matrixes */
  mtype *block_C = calloc(block_size * block_size, sizeof(*block_C));
  if (!block_C) {
    if (!rank)
      fputs("Can't allocate the memory for the result matrix!", stderr);
    MPI_Finalize();
    exit(1);
  }
  mtype temp_A;

  double start, end;

  start = MPI_Wtime();

  for (int i = 0; i < block_size; ++i) {
    for (int k = 0; k < block_size; ++k) {
      temp_A = block_A[i * block_size + k];
      for (int j = 0; j < block_size; ++j) {
        block_C[i * block_size + j] += temp_A * block_B[k * block_size + j];
      }
    }
  }

  end = MPI_Wtime();

  double time = end - start;
  double max_time;
  handle_mpi(
      MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD),
      "Error in defining the elapsed time");

  free(block_A);
  free(block_B);

  /* Creating new topo to perform summation along the axis */
  MPI_Comm line;
  int remain_dims[DIM_NUM] = {0, 0, 1};
  handle_mpi(MPI_Cart_sub(cube, remain_dims, &line),
             "Can't create a new topo {line}!");

  mtype *rec = malloc(block_size * block_size * sizeof(mtype));
  if (!rec) {
    if (!rank) {
      fputs("Can't allocate the necessary memory!", stderr);
    }
    MPI_Finalize();
    exit(1);
  }

  handle_mpi(MPI_Reduce(block_C, rec, block_size * block_size, MPI_INT32_T,
                        MPI_SUM, 0, line),
             "Can't sum along the axis!");
  free(block_C);

  print_result(argv[OUTPUT_FILE], cube, rec);

  if (!rank) {
    FILE *fg = fopen(argv[3], "rb");
    mtype *buf = malloc(sizeof(mtype) * n * n);
    fseek(fg, sizeof(mtype), SEEK_SET);
    fread(buf, sizeof(mtype), n * n, fg);
    for (int i = 0; i < n; printf("\n"), ++i)
      for (int j = 0; j < n; printf("%d ", buf[i * n + j]), ++j)
        ;
    fclose(fg);
    free(buf);
  }

  free(rec);
  handle_mpi(MPI_Comm_free(&line), "Can't delete topo {line}!");
  handle_mpi(MPI_Comm_free(&cube), "Can't delete topo {cube}!");

  if (!rank) {
    printf("\nElapsed time: %lf\n", max_time);
  }

  MPI_Finalize();
}
