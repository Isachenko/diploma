#include <stdio.h>
//#include <mpi.h>
#include <math.h>
#include <algorithm>
#include <string>

double l = 1.0;
int N = 3;
int N1 = N + 1;
int N1_2 = N1*N1;
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
  printf("N set to %d\n", N);
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

void sweep(double **A, double* b, double* y)
{
  //forward
  for (int line = 0; line < N1_2 - 1; ++line) {
    double main = A[line][N1];
    int maxI = std::min(N1, N1_2 - line - 1);
    for (int i = 1; i <= maxI; ++i) {
      double coef = A[line + i][N1 - i] / main;
      int maxJ = std::min(N1, N1_2 - line - 1);
      for (int j = 1; j <= maxJ; ++j) {
        A[line + i][N1 - i + j] -= coef * A[line][N1 + j];
      }
      b[line + i] -= coef * b[line];
    }
  }

  //backward
  for (int line = N1_2 - 1; line >= 0; --line) {
    double sum = 0;
    double maxI = std::min(N1, N1_2 - line - 1);
    for (int i = 1; i <= maxI; ++i) {
      sum += y[line + i]*A[line][N1 + i];
    }
    y[line] = (b[line] - sum) / A[line][N1];
  }
}

int main(int argc, char **argv)
{
  int p, rank;
  //MPI_Init(&argc, &argv);
  //MPI_Comm_size(MPI_COMM_WORLD, &p);
  //MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (argc > 1) {
    int n = atoi(argv[1]);
    reinit_N(n);
  }
  printf("start\n");

  double* y = new double[N1_2];
  for (int i1 = 0; i1 < N1; ++i1) {
    for (int i2 = 0; i2 < N1; ++i2) {
      y[i1*N1 + i2] = u0(i1, i2);
    }
  }

  double** A = zeros(N1_2, 2*N1 + 1);
  double* b = zeros(N1_2);

  int j = 1;
  while (t(j) < tmax) {
    for (int i = 0; i < N1_2; ++i) {
      std::fill(A[i], A[i] + (2*N1 + 1), (double)0);
    }
    std::fill(b, b + N1_2, (double)0);

    for(int i1 = 0; i1 < N1; ++i1) {
      for(int i2 = 0; i2 < N1; ++i2) {
        int line_index = i1*N1 + i2;
        double cn, dw, up, lf, rg, fi;

        get_coefs_and_fi(i1, i2, j, y[line_index], cn, dw, up, lf, rg, fi);

        A[line_index][N1] = cn;
        if ((line_index - 1) >= 0) {
          A[line_index][N1-1] = dw;
        }
        if ((line_index + 1) < N1_2) {
          A[line_index][N1+1] = up;
        }
        if ((line_index - N1) >= 0) {
          A[line_index][0] = lf;
        }
        if ((line_index + N1) < N1_2){
          A[line_index][2*N1] = rg;
        }
        b[line_index] = -fi;
      }
    }
    sweep(A,b,y);
    if (j == 1) {
      //break;
    }

    ++j;
  }

  //print answer
  // printf("t: %lf\n", t(j));
  // for (int i = 0; i < N1; ++i) {
  //   for (int j = 0; j < N1; ++j) {
  //     printf("%lf ", y[i * N1 + j]);
  //   }
  //   printf("\n");
  // }
  // printf("t: %lf\n", t(j));
  // for(int i = 0; i < N1_2; ++i) {
  //   // if (((i%4) / 2) == 0) {
  //   //   continue;
  //   // }
  //   for(int j = 0; j < 2*N1 + 1; ++j) {
  //     printf("%lf ", A[i][j]);
  //   }
  //   printf(" %lf", b[i]);
  //   printf("\n");
  //   // if (i % 4 == 3) {
  //   //   printf("\n");
  //   // }
  // }
  printf("finish\n");
  //MPI_Finalize();
  return(0);
}
