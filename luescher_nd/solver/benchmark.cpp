#include "solver.hpp"
#include <random>

void construct(
    ivec &rows,
    ivec &cols,
    dvec &coeffs,
    const size_t order,
    const float density
){
    const size_t entries(order*order*density);

    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_int_distribution<> randint(0, order-1);
    std::normal_distribution<> normal(0, 1);

    for(size_t i=0; i<entries; ++i){
        rows.push_back(randint(e2));
        cols.push_back(randint(e2));
        coeffs.push_back(normal(e2));
    }
}

int main(int argc, char** argv){
    ivec rows, cols;
    dvec coeffs;

    size_t order(10000);
    float density(0.05);
    int n_eigs(20);

    std::cout << "Order: " << order << std::endl;
    std::cout << "Density: " << density << std::endl;
    std::cout << "n_eigs: " << n_eigs << std::endl;

    std::cout << "Consructing sparse" << std::endl;
    construct(rows, cols, coeffs, order, density);
    std::cout << "Extracting eigs" << std::endl;
    const dvec eigs(get_eigs(n_eigs, order, rows, cols, coeffs));
    for(double eig : eigs ){
        std::cout << eig << ", ";
    }
    std::cout << std::endl;
}
