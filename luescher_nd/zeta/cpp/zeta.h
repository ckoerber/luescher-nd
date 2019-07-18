#ifndef ND_ZETA
#define ND_ZETA

#include <set>
#include <vector>
#include <map>
#include <math.h>
#include <complex>
#include <algorithm>
#include <functional>
#include <iostream> // Debugging

#ifdef CUBATURE
#include <cubature.h>
#endif


inline unsigned int nSquared(const std::vector<unsigned int> &v){
    unsigned int total=0;
    for(auto i:v) total+=i*i;
    return total;
}

class domain {
public:
    virtual ~domain() = default;
    virtual bool            filter(std::vector<unsigned int> &v) = 0;
    virtual unsigned int    degeneracy(std::vector<unsigned int> &v) = 0;
    virtual double          counterterm(const double &x) = 0;
    std::vector< std::vector<unsigned int> > vectors() { return this->vecs; };
    std::vector< unsigned int > degeneracies() {return this->degens; };
protected:
    unsigned int D; // Number of dimensions
    unsigned int N; // Diameter of each dimension
    bool improved;  // Improved?

    std::vector< std::vector<unsigned int> > enumerator();
    std::vector< std::vector<unsigned int> > vecs;
    std::vector< unsigned int > degens;
};

// THE WHOLE POINT:
double Zeta(domain &dom, double x);
std::vector<double> ZetaVectorized(domain &dom, const std::vector<double> &x);


// And now, details about particular domains:
class spherical:public domain{
public:
    spherical(const unsigned int D, const unsigned int N, bool improved=true);
    ~spherical() override = default;
    bool            filter(std::vector<unsigned int> &v) override;
    unsigned int    degeneracy(std::vector<unsigned int> &v) override;
    double          counterterm(const double &x) override;
protected:
    unsigned int radius; // What should the cutoff be?
};

class cartesian:public domain{
public:
    cartesian(const unsigned int D, const unsigned int N, bool improved=false);
    ~cartesian() override = default;
    bool            filter(std::vector<unsigned int> &v) override;
    unsigned int    degeneracy(std::vector<unsigned int> &v) override;
    double          counterterm(const double &x) override;

    // int integrator(unsigned ndim, const double *x, void *fdata, unsigned fdim, double *fval);
    // A stand-alone function in the .cc provides this functionality instead,
    // because I was getting an annoying error to do with cubature being unhappy getting
    // a non-static function pointer.
};

// The dispersion domain is the same as the cartesian, except for the counterterm.
class dispersion:public cartesian{
public:
    dispersion(const unsigned int D, const unsigned int N, const double L, const unsigned int nstep=1, bool improved=false);
    ~dispersion() override = default;
    double          counterterm(const double &x) override;

protected:
    double  L;
    unsigned int nstep;
    double  omega();

// private:     // In these I will implement different dispersion relations.
//     double  omega_1();
//     double  omega_2();
//     double  omega_3();
//     double  omega_4();
};




#endif
