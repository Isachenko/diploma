EXECS=matrix_swift_parallel
MPICC?=mpicxx

all: ${EXECS}

matrix_swift_parallel: matrix_swift_parallel.cpp
	${MPICC} -std=c++11 -o matrix_swift_parallel matrix_swift_parallel.cpp

clean:
	rm -f ${EXECS}
