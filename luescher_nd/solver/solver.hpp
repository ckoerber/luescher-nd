#include <Eigen/Core>
#include <vector>
#include <algorithm>
#include <Eigen/SparseCore>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>
#include <iostream>

using namespace Spectra;

typedef std::vector<double> dvec;
typedef std::vector<long> ivec;

typedef Eigen::Triplet<double> Triplet;

Eigen::SparseMatrix<double> convert_matrix(
    const long n,
    const ivec &rows,
    const ivec &cols,
    const dvec &coeffs
);
dvec get_eigs(
    const int nev,
    const long nmat,
    const ivec &rows,
    const ivec &cols,
    const dvec &coeffs
);
