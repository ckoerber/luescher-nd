#include "zeta.h"

using namespace std;

#define PI 3.141592653589793238463
#define RANGE(x, start, end) unsigned int x=start; x<end; x++

double Zeta(domain &dom, double x){
    double sum = 0.;
    auto vector = dom.vectors();
    auto degeneracy = dom.degeneracies();
    for(unsigned int i=0; i<vector.size(); i++){
        sum += degeneracy[i] / ( x - nSquared(vector[i]) );\
        // Probably needs to be replaced with dom.term(x, v) or some such,
        // To handle the dispersion case when it's some nstep-dependent function.
    }
    return sum + dom.counterterm(x);
}

inline unsigned int square(unsigned int x){
    return x*x;
}

inline unsigned int square_root(unsigned int x){
    return static_cast<unsigned int>(sqrt(x)) ; // Lossy!
}

int total(vector<int> v){
    int t=0;
    for(auto i:v) t+=i;
    return t;
}

std::vector< std::vector<unsigned int> > domain::enumerator(){
    // Produce all sorted, primitive d-dimensional integer vectors
    auto squares = vector<unsigned int>();
    for(RANGE(x, 0, 1+(this->radius))){
        squares.push_back(square(x));
    }

    // Build a list of squares that we'll increase dimension-by-dimension.
    auto vs = set< vector<unsigned int> >();

    // Initialize with vectors of desired squares,
    // essentially doing the 1D process:
    for(auto s:squares){
        auto temp = vector<unsigned int>();
        temp.push_back(s);
        vs.insert(temp);
    }

    // Now, go beyond 1D, building up by dimension:
    for(unsigned int dimension=2; dimension <= this->D; dimension++){
        auto next = set<vector<unsigned int> >();
        for(auto v:vs){
            for(auto s:squares){
                auto temp = v;
                temp.push_back(s);
                // NOTE: STILL SQUARED!
                if(this->filter(temp)){
                    sort(temp.begin(), temp.end());
                    next.insert(temp);
                }
            }
        }
        vs = next;
    }

    // De-square-root:
    auto no_sqrt = vector<vector<unsigned int> >();
    for(auto v:vs){
        vector<unsigned int> temp;
        for(auto i:v) temp.push_back(square_root(i));
        no_sqrt.push_back(temp);
    }
    sort(no_sqrt.begin(), no_sqrt.end(),
        [](vector<unsigned int> &a, vector<unsigned int> &b){
            return nSquared(a) <= nSquared(b);
        });

    return no_sqrt;
}

//////
//////      SPHERICAL DOMAIN
//////
spherical::spherical(const unsigned int DD, const unsigned int NN, bool iimproved){
    this->D = DD;
    this->N = NN;
    this->improved = iimproved;
    this->radius = NN/2;
    this->vecs = domain::enumerator();
    for(auto v:vecs) this->degens.push_back(this->degeneracy(v));
}

bool spherical::filter(std::vector<unsigned int> &squares){
    // Input is still squared!
    unsigned int t = 0;
    for(auto s:squares) t+=s;
    return t <= square(this->radius);
}

unsigned int spherical::degeneracy(std::vector<unsigned int> &v){
    // ASSUMES VECTOR IS SORTED!
    if(this->D==1){
        if(v[0] == 0) return 1;
        return 2;
    }
    if(this->D==2){
        if(v[0] == 0 && v[1] == 0)  return 1;
        if(v[0] == 0 || v[1] == 0)  return 4;
        if(v[0] == v[1])            return 4;
        return 8;
    }
    if(this->D==3){
        if(v[0] == 0 && v[1] == 0 && v[2] == 0) return 1;
        if(v[0] == 0 && v[1] == 0)              return 6;
        if(v[0] == 0 &&      v[1] == v[2])      return 12;
        if(v[0] == 0 && v[1] !=      v[2])      return 24;
        if(v[0] == v[1] && v[0] ==   v[2])      return 8;
        if(v[0] ==      v[1])                   return 24;
        if(             v[1] ==      v[2])      return 24;
        return 48;
    }
    // Beyond d=3 I will not implement.
    return 0;
}

double spherical::counterterm(const double &x){
    double ct = 0;

    // counterterm
    if(this->D==1) ct+=0; // FIX
    if(this->D==1) ct+=0; // FIX
    if(this->D==3) ct+= +4*PI*this->radius;

    if(this->improved){
        // We have to take a detour through the complex plane.
        // Miraculously, the imaginary parts cancel!
        complex<double> sqrt_x = sqrt(complex<double>(x)); // TODO: fix
        if(this->D==1)  ct+= 0.; // FIX
        if(this->D==2)  ct+= 0.; // FIX
        if(this->D==3)  ct+= -4*PI*real(sqrt_x * atanh( sqrt_x / double((this->N)/2) ));
    }
    return ct;
}

//////
//////      CARTESIAN DOMAIN
//////

// TODO


//////
//////      DISPERSION DOMAIN
//////

// TODO
