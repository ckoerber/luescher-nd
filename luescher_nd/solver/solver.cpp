#include <Eigen/Core>
#include <vector>
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
){
    Eigen::SparseMatrix<double> M(n, n);
    std::vector<Triplet> triplet_list;

    for(size_t i = 0; i < rows.size(); ++i){
        triplet_list.push_back(Triplet(rows[i], cols[i], coeffs[i]));
    }
    M.setFromTriplets(triplet_list.begin(), triplet_list.end());
    return M;
}

void convert_print_mat(
    const long n,
    const ivec &rows,
    const ivec &cols,
    const dvec &coeffs
){
    std::cout << convert_matrix(n, rows, cols, coeffs) << std::endl;
}

int main()
{
    // A band matrix with 1 on the main diagonal, 2 on the below-main subdiagonal,
    // and 3 on the above-main subdiagonal
    const int n = 10;
    Eigen::SparseMatrix<double> M(n, n);
    M.reserve(Eigen::VectorXi::Constant(n, 3));
    for(int i = 0; i < n; i++)
    {
        M.insert(i, i) = 1.0;
        if(i > 0)
            M.insert(i - 1, i) = 3.0;
        if(i < n - 1)
            M.insert(i + 1, i) = 2.0;
    }

    // Construct matrix operation object using the wrapper class SparseGenMatProd
    SparseGenMatProd<double> op(M);

    // Construct eigen solver object, requesting the largest three eigenvalues
    GenEigsSolver<double, SMALLEST_MAGN, SparseGenMatProd<double> > eigs(&op, 3, 6);

    // Initialize and compute
    eigs.init();
    int nconv = eigs.compute();

    // Retrieve results
    Eigen::VectorXcd evalues;
    if(eigs.info() == SUCCESSFUL)
        evalues = eigs.eigenvalues();

    std::cout << "Eigenvalues found:\n" << evalues << std::endl;

    return 0;
}
