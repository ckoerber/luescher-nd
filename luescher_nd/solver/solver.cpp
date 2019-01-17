#include "solver.hpp"


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

dvec get_eigs(
    const int nev,
    const long nmat,
    const ivec &rows,
    const ivec &cols,
    const dvec &coeffs
){
    const Eigen::SparseMatrix<double> M(convert_matrix(nmat, rows, cols, coeffs));
    SparseGenMatProd<double> op(M);
    GenEigsSolver<double, SMALLEST_REAL, SparseGenMatProd<double>> eigs(&op, nev, std::max(2*nev, nev+2));

    eigs.init();
    eigs.compute();

    Eigen::VectorXd evalues;
    if(eigs.info() != SUCCESSFUL){}

    evalues = eigs.eigenvalues().real();
    dvec vec(evalues.data(), evalues.data() + evalues.size());

    return vec;
}
