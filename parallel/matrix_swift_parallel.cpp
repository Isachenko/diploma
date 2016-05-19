#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <algorithm>
#include <string>

//default initialization
double l = 1.0;
int N = 30;
int N1 = N + 1;
int N1_2 = N1*N1;
int B_INDEX = 2*N1 + 1; // vectro b is stored inside matrix A as last column
int ROW_SIZE = 2*N1 + 2;
double h = l / N;

double tmax = 1;
double tau = 0.005;

double alph1 = 1;
double alph2 = 1;
double alph3 = 1;
double alph4 = 1;

double beta1 = 1;
double beta2 = 1;
double beta3 = 1;
double beta4 = 1;

auto x = [](int i){return i*h;};
auto t = [](int j){return j*tau;};

auto i_to_x_sum = [](int i1, int i2, int j){return x(i1) + x(i2) + t(j);};

auto u_real = [](int i1, int i2, int j){return exp(i_to_x_sum(i1, i2, j));};
auto du_real = [](int i1, int i2, int j){return exp(i_to_x_sum(i1, i2, j));};
auto ddu_real = [](int i1, int i2, int j){return exp(i_to_x_sum(i1, i2, j));};

auto k1 = [](int i1, int i2, int j){return 1;/*exp(i_to_x_sum(i1, i2, j));*/};
auto dk1 = [](int i1, int i2, int j){return 0;/*exp(i_to_x_sum(i1, i2, j))}*/};

auto k2 = [](int i1, int i2, int j){return 1;/*exp(i_to_x_sum(i1, i2, j));*/};
auto dk2 = [](int i1, int i2, int j){return 0;/*exp(i_to_x_sum(i1, i2, j));*/};

auto u0 = [](int i1, int i2){return exp(i_to_x_sum(i1, i2, 0));};

auto gamma1 = [](int i2, int j){return beta1*u_real(0, i2, j) + alph1*k1(0,i2,j)*du_real(0, i2, j);};
auto gamma2 = [](int i2, int j){return beta2*u_real(N, i2, j) - alph2*k1(N,i2,j)*du_real(N, i2, j);};
auto gamma3 = [](int i1, int j){return beta3*u_real(i1, 0, j) + alph3*k2(i1,0,j)*du_real(i1, 0, j);};
auto gamma4 = [](int i1, int j){return beta4*u_real(i1, N, j) - alph4*k2(i1,N,j)*du_real(i1, N, j);};

auto f = [](int i1, int i2, int j) {
  return du_real(i1,i2,j) - (dk1(i1,i2,j)*du_real(i1,i2,j) + k1(i1,i2,j)*ddu_real(i1,i2,j)) -
  (dk2(i1,i2,j)*du_real(i1,i2,j) + k2(i1,i2,j)*ddu_real(i1,i2,j));
};

auto a1 = [](int i1, int i2, int j){return (k1(i1, i2, j) + k1(i1 - 1, i2, j)) / 2.0;};
auto a2 = [](int i1, int i2, int j){return (k2(i1, i2, j) + k2(i1, i2 - 1, j)) / 2.0;};

void reinit_N(int n) {
  N = n;
  N1 = N + 1;
  N1_2 = N1*N1;
  h = l / N;
  B_INDEX = 2*N1 + 1;
  ROW_SIZE = 2*N1 + 2;
  //printf("N set to %d\n", N);
}

void get_coefs_and_fi(const int &i1, const int &i2 , const int &j, const double &y_prev,
                      double &ct, double &dw, double &up, double &lf, double &rg, double &fi)
{
  ct = 1 / tau;
  dw = 0;
  up = 0;
  lf = 0;
  rg = 0;
  fi = (-y_prev / tau) - (f(i1,i2,j));

  //by x1
  if (i1 == 0) {
    ct += (2*a1(i1+1,i2,j) / (h*h)) - (2*beta1 / (h*alph1));
    rg += -2*a1(i1+1,i2,j) / (h*h);
    lf = 0;
    fi += (2*gamma1(i2,j)) / (h*alph1);
  }
  else if (i1 == N) {
    ct += (2*a1(i1,i2,j) / (h*h)) - (2*beta2 / (h*alph2));
    rg = 0;
    lf += -2*a1(i1,i2,j) / (h*h);
    fi += (2*gamma2(i2,j)) / (h*alph2);
  }
  else {
    ct += (a1(i1+1,i2,j) + a1(i1,i2,j)) / (h*h);
    lf += -a1(i1,i2,j) / (h*h);
    rg += -a1(i1+1,i2,j) / (h*h);
  }

  //by x2
  if (i2 == 0) {
    ct += (2*a2(i1,i2+1,j) / (h*h)) - (2*beta3 / (h*alph3));
    up += -2*a2(i1,i2+1,j) / (h*h);
    dw = 0;
    fi += (2*gamma3(i1,j)) / (h*alph3);
  }
  else if (i2 == N) {
    ct += (2*a2(i1,i2,j) / (h*h)) - (2*beta4 / (h*alph4));
    up = 0;
    dw += -2*a2(i1,i2,j) / (h*h);
    fi += (2*gamma4(i1,j)) / (h*alph4);
  }
  else {
    ct += (a2(i1,i2+1,j) + a2(i1,i2,j)) / (h*h);
    dw += -a2(i1,i2,j) / (h*h);
    up += -a2(i1,i2+1,j) / (h*h);
  }
}

double* zeros(int n) {
  double *arr = new double[n];
  std::fill(arr, arr + n, (double)0);
  return arr;
}

double** zeros(int n1, int n2) {
  double** arr = new double*[n1];
  for (int i = 0; i < n1; ++i) {
    arr[i] = zeros(n2);
  }
  return arr;
}

void sweep(double **A, double* y,
            const int &mpi_size, const int &mpi_rank)
{
  int block_size = N1 / mpi_size;

  //forward
  double* main_line = nullptr;
  double* local_buffer = new double[N1 + 2];
  int i_begin = -1;
  int offset = -1;
  int max_i = -1;

  //printf("here\n");
  for (int i1 = 0; i1 < N1; ++i1) {
    for (int root = 0; root < mpi_size; ++root) {
      for (int index = 0; index < block_size; ++index) {
        //communication
        if (mpi_rank == root) {
          int local_index = i1*block_size + index;
          main_line = A[local_index] + N1;
          i_begin = local_index + 1;
          offset = 1;
          max_i = std::min((i1+1)*(block_size), N1*block_size);
        } else {
          main_line = local_buffer;
          i_begin = i1*block_size + (mpi_rank > root ? 0 : block_size);
          offset = (mpi_rank > root ? mpi_rank - root : mpi_size - (root - mpi_rank)) * block_size - index;
          max_i = std::min(i_begin + block_size, N1*block_size);
        }
        MPI_Bcast((void*)(main_line), N1 + 2, MPI_DOUBLE, root, MPI_COMM_WORLD);
        double main = main_line[0];
        //printf("%lf %d\n", main, mpi_rank);
        if (mpi_rank == 1) {
          //printf("%d %d %d\n", offset, i_begin, max_i);
        }

        //calculation
        //int global_line = i1*N1 + root*block_size + index;
        //calc for local lines
        for (int i = i_begin; i < max_i; ++i) {
          int delta_i = (i - i_begin) + offset;
          double coef = A[i][N1 - delta_i] / main;
          if (mpi_rank == 1) {
            //printf("%d\n", i);
          }
          for (int j = 1; j <= N1; ++j) {
            //printf("%lf, %lf, %lf, %lf\n", A[i][N1 - delta_i + j], A[i][N1 - delta_i], main_line[j], main);
            A[i][N1 - delta_i + j] -= coef * main_line[j];
            // if (mpi_rank == 1) {
            //   printf("yzi: %d %d\n", mpi_rank, N1 - delta_i + j);
            // }
          }
          //printf("%lf %d\n", main_line[B_INDEX], B_INDEX);
          A[i][B_INDEX] -= coef * main_line[B_INDEX - N1];
        }
        //calc for far lines if root
        if (mpi_rank == root) {
          offset = N1 - index;
          i_begin = (i1+1)*block_size;
          max_i = std::min(i_begin + index + 1, N1*block_size);
          if (mpi_rank == 1) {
            //printf("%d %d %d %d\n", mpi_rank, i_begin, max_i, offset);
          }
          for (int i = i_begin; i < max_i; ++i) {
            int delta_i = (i - i_begin) + offset;
            double coef = A[i][N1 - delta_i] / main;
            for (int j = 1; j <= N1; ++j) {
              //printf("%lf, %lf, %lf, %lf\n", A[i][N1 - delta_i + j], A[i][N1 - delta_i], main_line[j], main);
              A[i][N1 - delta_i + j] -= coef * main_line[j];
              // if (mpi_rank == 1) {
              //   printf("yzi: %d %d\n", mpi_rank, N1 - delta_i + j);
              // }
            }
            A[i][B_INDEX] -= coef * main_line[B_INDEX - N1];
          }
        }

      }
    }
  }

  //backward
  double* sum = new double[block_size * N1];
  std::fill(sum, sum + block_size * N1, (double)0);

  for (int i1 = N1 - 1; i1 >= 0; --i1) { // for every big block
    for (int root = mpi_size - 1; root >= 0; --root) {  // for every small block
      if (mpi_rank == root) { //if it is our block
        //calculate local y's
        int offset = (i1+1)*(block_size) - 1; // last one is here

        if (mpi_rank == 1) {
          //printf("%d procees in %d %d BEFORE: %lf, %lf\n",mpi_rank, i1, root, y[1 + offset - block_size],y[2 + offset - block_size]);
        }
        for (int i = offset; i > offset - block_size; --i) { //for every y from small block
          y[i] = (A[i][B_INDEX] - sum[i]) / A[i][N1]; //calc cuurent y
          if (mpi_rank == 1) {
            //printf("params: %d %lf %lf\n", i, sum[i], A[i][N1]);
          }
          for (int j = i - 1; j > offset - block_size; --j) { // and count him for cur block
          if (mpi_rank == 1) {
            //printf("%d %d sum1 before: %lf\n", i1, root,sum[j]);
          }
            sum[j] += y[i] * A[j][N1 + (i-j)];

            if (mpi_rank == 1) {
              //printf("%d %d sum1 after: %lf\n", i1, root,sum[j]);
            }
          }
          //count for next big block
          if (i1 > 0) {
            int next_offset = i1*(block_size) - 1;
            for (int j = next_offset; j >= next_offset - (offset - i); --j) {
            if (mpi_rank == 1) {
              //printf("%d %d sum2 before: %lf\n", i1, root,sum[j]);
            }
              sum[j] += y[i] * A[j][N1 + (i-j) + ((mpi_size-1)*block_size)];
              if (mpi_rank == 1) {
                //printf("%d %d sum2 after: %lf\n", i1, root,sum[j]);
              }
            }
          }
        }
        // //then send local y's to other proceses

        //printf("%d procees in %d %d SEND: %lf %d\n",mpi_rank, i1, root, y[1 + offset - block_size], offset);


        int mpi_err = MPI_Bcast((void*)(y + (1 + offset - block_size)), block_size, MPI_DOUBLE, root, MPI_COMM_WORLD);
        //printf("%d procees in %d %d SENT: %lf\n",mpi_rank, i1, root, y[1 + offset - block_size]);

        if (mpi_err != MPI_SUCCESS) {
          printf("probleme %d\n", mpi_err);
        }
      } else {
        //get y's from other process
        //printf("%d procees in %d %d RCV: %lf\n",mpi_rank, i1, root, local_buffer[0]);

        MPI_Bcast((void*)(local_buffer), block_size, MPI_DOUBLE, root, MPI_COMM_WORLD);
        //printf("%d procees in %d %d RCVED: %lf\n",mpi_rank, i1, root, local_buffer[0]);


        if (mpi_rank == 1) {
          //printf("%d procees in %d %d RCV: %lf, %lf\n",mpi_rank, i1, root, local_buffer[0],local_buffer[1]);
        }
        //and count them for ours lines
        if ((i1 > 0) || (mpi_rank < root)) {
          int offset = (i1+1)*block_size - (mpi_rank < root ? 0 : block_size) - 1;
          int diff_offset = (((mpi_rank > root ? (mpi_size) - (mpi_rank - root) : root - mpi_rank)-1) * block_size) + 1;
          for (int i = offset; i > offset - block_size; --i) {
            for (int j = 0; j < block_size; ++j) { // for each gotten y
              // if (mpi_rank == 0) {
              //   printf("%d %d: diff_offset: %d, %d\n",i, j, diff_offset, N1 + diff_offset + j);
              // }
              if (mpi_rank == 1) {
                //printf("%d %d sum3 before: %lf\n", i1, root,sum[i]);
              }
              sum[i] += local_buffer[j] * A[i][N1 + diff_offset + j];
              if (mpi_rank == 1) {
                //printf("%d %d sum3 after: %lf\n", i1, root,sum[i]);
              }
            }
            diff_offset += 1;
          }
        }
      }
    }
  }
  delete[] local_buffer;
  delete[] sum;
}

//works only if N % p = 0
void init_matrix(double **A, double *y,
                const int &j, const int &mpi_size, const int &mpi_rank) {
  int block_size = N1 / mpi_size;
  int begin_i2 = block_size * mpi_rank;
  int end_i2 = block_size * (mpi_rank + 1);
  for(int i1 = 0; i1 < N1; ++i1) {
    for(int i2 = begin_i2; i2 < end_i2; ++i2) {
      int global_line_index = i1*N1 + i2;
      int line_index = i1*block_size + (i2 - begin_i2);

      double cn, dw, up, lf, rg, fi;

      get_coefs_and_fi(i1, i2, j, y[line_index], cn, dw, up, lf, rg, fi);

      A[line_index][N1] = cn;
      if ((global_line_index - 1) >= 0) {
        A[line_index][N1-1] = dw;
      }
      if ((global_line_index + 1) < N1_2) {
        A[line_index][N1+1] = up;
      }
      if ((global_line_index - N1) >= 0) {
        A[line_index][0] = lf;
      }
      if ((global_line_index + N1) < N1_2){
        A[line_index][2*N1] = rg;
      }
      A[line_index][B_INDEX] = -fi;

    }
  }
}

int main(int argc, char **argv)
{
  //init mpi
  int mpi_size, mpi_rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  //init params
  if (argc > 1) {
    int n = atoi(argv[1]);
    reinit_N(n);
  }
  if (mpi_rank == 0) {
    printf("N set to %d\n", N);
    printf("start\n");
  }

  //init start values
  int block_size = N1 / mpi_size;
  int begin_i2 = block_size * mpi_rank;
  int end_i2 = block_size * (mpi_rank + 1);
  double* y = new double[N1_2 / mpi_size];
  for(int i1 = 0; i1 < N1; ++i1) {
    for(int i2 = begin_i2; i2 < end_i2; ++i2) {
      y[i1*block_size + (i2 - begin_i2)] = u0(i1, i2);
    }
  }
  double** A = zeros(N1_2 / mpi_size, ROW_SIZE);
  //
  int j = 1;
  while (t(j) < tmax) {
    //refresh matrix
    for (int i = 0; i < N1_2 / mpi_size; ++i) {
      std::fill(A[i], A[i] + ROW_SIZE, (double)0);
    }

    init_matrix(A, y, j, mpi_size, mpi_rank);

    sweep(A, y, mpi_size, mpi_rank);
    if (j == 1) {
      //break;
    }
    ++j;
  }

  //print answer
  // if (mpi_rank == 3) {
  //   printf("t: %lf\n", t(j));
  //   for(int i1 = 0; i1 < N1; ++i1) {
  //     for(int i2 = begin_i2; i2 < end_i2; ++i2) {
  //       printf("%lf ",y[i1*block_size + (i2 - begin_i2)]);
  //     }
  //     printf("\n");
  //   }
  // }
  //print matrix
  // if (mpi_rank == 1) {
  //   for(int i = 0; i < N1*block_size; ++i) {
  //     for(int j = 0; j < 2*N1 + 2; ++j) {
  //       printf("%lf ", A[i][j]);
  //     }
  //     printf("\n");
  //   }
  // }


  //clean
  // delete[] b;
  // delete[] y;
  // for (int i = 0; i < N1; ++i) {
  //   delete[] A[i];
  // }
  // delete A;

  if (mpi_rank == 0){
    printf("finish\n");
  }
  MPI_Finalize();
  return(0);
}
