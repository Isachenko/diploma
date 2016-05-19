EXECS=matrix_swift
MPICC?=mpicxx

all: ${EXECS}

matrix_swift: matrix_swift.cpp
	${MPICC} -std=c++11 -o matrix_swift matrix_swift.cpp

clean:
	rm -f ${EXECS}
