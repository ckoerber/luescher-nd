#ifndef ND_ZETA
#define ND_ZETA

#include <set>
#include <vector>
#include <map>
#include <math.h>
#include <complex>
#include <algorithm>
#include <functional>

// struct domain {
//     std::function<bool(const std::vector<int>&)>        filter;
//     // std::function<int(const std::vector<int>&)>         degeneracy;
//     std::function<double(const int&, const double&)>    counterterm;
//     bool improved;
// };

inline unsigned int nSquared(std::vector<unsigned int> v){
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
    unsigned int radius; // What should the cutoff be?

    std::vector< std::vector<unsigned int> > enumerator();
    std::vector< std::vector<unsigned int> > vecs;
    std::vector< unsigned int > degens;
};

// THE WHOLE POINT:
double Zeta(domain &dom, double x);


// And now, details about particular domains:
class spherical:public domain{
public:
    spherical(const unsigned int D, const unsigned int N, bool improved=true);
    ~spherical() override = default;
    bool            filter(std::vector<unsigned int> &v);
    unsigned int    degeneracy(std::vector<unsigned int> &v);
    double          counterterm(const double &x);
};

class cartesian:public domain{
public:
    cartesian(const unsigned int D, const unsigned int N, bool improved=false);
    ~cartesian() override = default;
    bool            filter(std::vector<unsigned int> &v);
    unsigned int    degeneracy(std::vector<unsigned int> &v);
    double          counterterm(const double &x);
};

// To be implemented:
// class dispersion:public domain{
// public:
//     dispersion(const unsigned int D, const unsigned int N, bool improved=true);
//     ~dispersion() override = default;
//     bool            filter(std::vector<unsigned int> &v);
//     unsigned int    degeneracy(std::vector<unsigned int> &v);
//     double          counterterm(const double &x);
// }

#endif
