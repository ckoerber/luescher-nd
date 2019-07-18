#ifndef ND_ZETA_WRAPPER
#define ND_ZETA_WRAPPER

#include "zeta.h"

class SphericalZeta_CC{
public:
    SphericalZeta_CC(
        const unsigned int D, const unsigned int N, bool improved=true
    ) : dom(spherical(D, N, improved)) {};
    ~SphericalZeta_CC() = default;
    double operator()(const double &x){return Zeta(this->dom, x); };
    std::vector<double> operator()(const std::vector<double> &x){return ZetaVectorized(this->dom, x);};

private:
    spherical dom;
};

class CartesianZeta_CC{
public:
    CartesianZeta_CC(
        const unsigned int D, const unsigned int N, bool improved=true
    ) : dom(cartesian(D, N, improved)) {};
    ~CartesianZeta_CC() = default;
    std::vector<double> operator()(const std::vector<double> &x){return ZetaVectorized(this->dom, x);};

private:
    cartesian dom;
};

class DispersionZeta_CC{
public:
    DispersionZeta_CC(
        const unsigned int D,
        const unsigned int N,
        const double L,
        const unsigned int nstep=1,
        bool improved=false
    ) : dom(dispersion(D, N, L, nstep, improved)) {};
    ~DispersionZeta_CC() = default;
    std::vector<double> operator()(const std::vector<double> &x){return ZetaVectorized(this->dom, x);};

private:
    dispersion dom;
};

#endif
