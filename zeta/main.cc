#include "zeta.h"
#include <stdio.h>
#include <algorithm>
#include <iostream>

int main(){
    unsigned int D=3;
    unsigned int N = 100;
    auto dom = spherical(D, N);

    for(auto v:dom.vectors()){
        printf("[%d %d %d] n^2= %d #= %d\n", v[0], v[1], v[2], nSquared(v), dom.degeneracy(v));
    }

    double x = 1.234;
    double s = Zeta(dom, x);
    printf("S(D=%d, N=%d, %f) = %.16f\n", D, N, x, s);

    double min=-4.;
    double max=10.;
    unsigned int steps=5000;
    double step = (max-min)/(steps-1);

    auto X = std::vector<double>(steps);
    auto S = std::vector<double>();
    std::generate(X.begin(), X.end(), [n=min, step]() mutable { return n+=step; } );
    std::transform(X.begin(), X.end(), std::back_inserter(S), [&](double xx){ return Zeta(dom, xx); });

    printf("c={");
    for(unsigned int i=0; i<X.size()-1; i++){
        printf("{%.2f, %.16f}, ", X[i], S[i]);
    }
    printf("{%.2f, %.16f}}; ", X[X.size()-1], S[X.size()-1]);

}
