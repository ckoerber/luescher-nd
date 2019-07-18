#include "zeta.h"
#include "zeta_wrapper.h"
#include <stdio.h>
#include <algorithm>
#include <iostream>

int main(){

    unsigned int D=3;
    unsigned int N = 10;

    printf("Setting up..."); fflush(stdout);
    auto sphere = spherical(D, N);
    auto sphere_unimproved = spherical(D, N, false);
    printf("done!\n");  fflush(stdout);

    // for(auto v:sphere.vectors()){
    //     printf("[%d %d %d] n^2= %d #= %d\n", v[0], v[1], v[2], nSquared(v), sphere.degeneracy(v));
    // }

    double x = 1.234;
    double s = Zeta(sphere, x);
    printf("spherical(D=%d, N=%d, %f) = %.16f\n", D, N, x, s);
    s = Zeta(sphere_unimproved, x);
    printf("spherical(D=%d, N=%d, %f) = %.16f unimproved\n", D, N, x, s);

    std::vector<double> xv(4);
    xv[0] = -8.901; xv[1] = 9.012; xv[2] = 0.123; xv[3] = 1.234;
    std::vector<double> sv = ZetaVectorized(sphere, xv);
    for(unsigned int i=0; i<sv.size(); i++){
        printf("spherical(D=%d, N=%d, %f) = %.16f vectorized\n", D, N, xv[i], sv[i]);
    }

    auto spherical_zeta = SphericalZeta_CC(D, N);
    s = spherical_zeta(x);
    printf("spherical(D=%d, N=%d, %f) = %.16f wrapped\n", D, N, x, s);

    auto box = cartesian(D, N, false);
    s = Zeta(box, x);
    printf("box(D=%d, N=%d, %f) = %.16f\n", D, N, x, s);

    auto box_improved = cartesian(D, N, true);
    s = Zeta(box_improved, x);
    printf("box_improved(D=%d, N=%d, %f) = %.16f\n", D, N, x, s);


    return(0);

    double min=-4.;
    double max=10.;
    unsigned int steps=500;
    double step = (max-min)/(steps-1);

    auto X = std::vector<double>(steps);
    auto S = std::vector<double>();
    auto B = std::vector<double>();
    std::generate(X.begin(), X.end(), [n=min, step]() mutable { return n+=step; } );
    std::transform(X.begin(), X.end(), std::back_inserter(S), [&](double xx){ return Zeta(sphere, xx); });
    std::transform(X.begin(), X.end(), std::back_inserter(B), [&](double xx){ return Zeta(box, xx); });

    printf("s[%d]={",N);
    for(unsigned int i=0; i<X.size()-1; i++){
        printf("{%.2f, %.16f}, ", X[i], S[i]);
    }
    printf("{%.2f, %.16f}};\n", X[X.size()-1], S[X.size()-1]);

    printf("b[%d]={",N);
    for(unsigned int i=0; i<X.size()-1; i++){
        printf("{%.2f, %.16f}, ", X[i], B[i]);
    }
    printf("{%.2f, %.16f}};\n", X[X.size()-1], B[X.size()-1]);


}
