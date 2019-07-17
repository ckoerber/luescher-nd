#include "zeta.h"


class SphericalZeta{
public:
    SphericalZeta(
        const unsigned int D, const unsigned int N, bool improved=true
    ) : dom(spherical(D, N, improved)) {};
    ~SphericalZeta() = default;
    double operator()(const double x){return Zeta(this->dom, x);};

private:
    spherical dom;
};

class CartesianZeta{
public:
    CartesianZeta(
        const unsigned int D, const unsigned int N, bool improved=true
    ) : dom(cartesian(D, N, improved)) {};
    ~CartesianZeta() = default;
    double operator()(const double x){return Zeta(this->dom, x);};

private:
    cartesian dom;
};

class DispersionZeta{
public:
    DispersionZeta(
        const unsigned int D,
        const unsigned int N,
        const double L,
        const unsigned int nstep=1,
        bool improved=false
    ) : dom(dispersion(D, N, L, nstep, improved)) {};
    ~DispersionZeta() = default;
    double operator()(const double x){return Zeta(this->dom, x);};

private:
    dispersion dom;
};
