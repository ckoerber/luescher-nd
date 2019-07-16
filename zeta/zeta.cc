#include "zeta.h"

using namespace std;

#define PI 3.141592653589793238463
#define RANGE(x, start, end) unsigned int x=start; x<end; x++

double Zeta(domain &dom, double x){
    double sum = 0.;
    auto vector = dom.vectors();
    auto degeneracy = dom.degeneracies();
    // int count=0;
    for(unsigned int i=0; i<vector.size(); i++){
        // printf("[%d %d %d] #=%d\n", vector[i][0], vector[i][1], vector[i][2], degeneracy[i]);
        // count++;
        sum += degeneracy[i] / ( nSquared(vector[i]) - x );\
        // Probably needs to be replaced with dom.term(x, v) or some such,
        // To handle the dispersion case when it's some nstep-dependent function.
    }
    // printf("%d terms\n", count);
    // NB: the counterterm gets SUBTRACTED.
    return sum - dom.counterterm(x);
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
    for(RANGE(x, 0, 1+(this->N/2))){
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

    // The following is NOT guaranteed to work by the C++ standard.
    // The instantiated function that overrides the virtual functions need
    // not be available in the vtable until the constructor is finished.
    // Since enumerator() calls this->filter() and the filter() is defined by
    // the inherited class, this is one example of undefined behavior.
    // However, it at least works with
    //      - g++-9 (Homebrew GCC 9.1.0) 9.1.0.
    //      - Apple LLVM version 10.0.1 (clang-1001.0.46.4)

    // It should be safe to make enumerator take a function pointer to a filter.
    // In the meantime, this is nice.
    this->vecs = this->enumerator();
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

    // We have to take a detour through the complex plane.
    // Miraculously, the imaginary parts cancel!
    // TODO: what about 2D?
    complex<double> z = complex<double>(x);
    complex<double> sqrt_z = sqrt(z);
    double N_half = double(this->radius);

    if(this->improved){
        // complete behavior
        if(this->D==1) ct= - 2 * real(atanh(sqrt_z/N_half) / sqrt_z);
        if(this->D==2) ct= + real(PI*log( N_half*N_half/z - 1.)); // TODO: is it right when z < 0?
        if(this->D==3) ct= + 4*PI*(N_half - real(atanh(sqrt_z/N_half) * sqrt_z));
    }
    else{
        // just behavior divergent in N.
        if(this->D==1)  ct = - 2 / N_half;
        if(this->D==2)  ct = + real(PI*log(N_half*N_half / z)); // TODO: is it right when z < 0?
        if(this->D==3)  ct = + 4*PI*N_half;
    }

    return ct;
}

//////
//////      CARTESIAN DOMAIN
//////

cartesian::cartesian(const unsigned int DD, const unsigned int NN, bool iimproved){
    this->D = DD;
    this->N = NN;
    this->improved = iimproved;
    this->vecs = domain::enumerator();
    for(auto v:vecs) this->degens.push_back(this->degeneracy(v));
}

bool cartesian::filter(std::vector<unsigned int> &squares){
    // Input is still squared!
    bool inCube = true;
    for(auto s:squares) inCube = inCube && s <= square(this->N/2);
    return inCube;
}

unsigned int cartesian::degeneracy(std::vector<unsigned int> &v){
    // ASSUMES VECTOR IS SORTED!
    unsigned int result = 0;
    if(this->D==1){
        if(v[0] == 0) result=1;
        else result=2;
    }
    if(this->D==2){
        if     (v[0] == 0 && v[1] == 0)  result=1;
        else if(v[0] == 0 || v[1] == 0)  result=4;
        else if(v[0] == v[1])            result=4;
        else result=8;
    }
    if(this->D==3){
        if     (v[0] == 0 && v[1] == 0 && v[2] == 0) result=1;
        else if(v[0] == 0 && v[1] == 0)              result=6;
        else if(v[0] == 0 &&      v[1] == v[2])      result=12;
        else if(v[0] == 0 && v[1] !=      v[2])      result=24;
        else if(v[0] == v[1] && v[0] ==   v[2])      result=8;
        else if(v[0] ==      v[1])                   result=24;
        else if(             v[1] ==      v[2])      result=24;
        else result=48;
    }
    // Beyond d=3 I will not implement.
    if(this->D>3) result=0;

    for(auto i:v){
        if(i==this->N/2) result/=2;
    }
    return result;
}

int cartesian_integrator(unsigned ndim, const double *nu, void *fdata, unsigned fdim, double *fval){
    double* params = static_cast<double *>(fdata);
    double N_half = params[0];
    double x = params[1];
    double numerator = 1.;
    double denominator=-x;
    for(unsigned int d=0; d < ndim; d++) denominator += nu[d] * nu[d];
    fval[0] = numerator/denominator;
    printf("N/2=%.2f x=%.10f %.10f/%.10f = %.10f ; [",N_half, x, numerator, denominator, fval[0]);
    for(unsigned int d=0; d < ndim; d++) cout << nu[d] << " ";
    cout << "]" << endl;
    return 0;
}

double cartesian::counterterm(const double &x){
    double ct = 0;
    double error = 0;
    double tolerance = 1e-2;
    error_norm e = ERROR_L1; // ERROR_{L1,L2,LINF,INDIVIDUAL,PAIRED}

    // We have to take a detour through the complex plane.
    // Miraculously, the imaginary parts cancel!
    // TODO: what about 2D?
    complex<double> z = complex<double>(x);
    complex<double> sqrt_z = sqrt(z);
    double N_half = double(this->N/2);

    if(this->improved){
        // In 1D the cartesian and spherical volumes are the same.
        if(this->D==1) ct= -2 * real(tanh(sqrt_z / N_half) / sqrt_z);
        if(this->D==2) {
            #ifdef CUBATURE
                // Integrate in one quadrant only.
                double nu_min[2] = {0,0}, nu_max[2] = {+1,+1};
                double xtwiddle = x / pow(N_half,2);
                double params[2] = {N_half, xtwiddle};
                hcubature(1, cartesian_integrator, params, D, nu_min, nu_max, 0, 0, tolerance, e, &ct, &error);
                // then multiply to cover the square.
                ct*=4;
                // then multiply in the scaling behavior
                // ct*=pow(N_half,this->D-2) // doesn't do anything in D=2
            #else
                ct=0; // FIX
            #endif
        }
        if(this->D==3) {
            #ifdef CUBATURE
                // Integrate in one octant only.
                double nu_min[3] = {0,0,0}, nu_max[3] = {+1,+1,+1};
                double xtwiddle = x / pow(N_half,2);
                double params[2] = {N_half, xtwiddle};
                hcubature(1, cartesian_integrator, params, D, nu_min, nu_max, 0, 0, tolerance, e, &ct, &error);
                // then multiply to cover the cube.
                ct *= 8;
                // then multiply in the scaling behavior
                ct*=pow(N_half,this->D-2); // doesn't do anything in D=2
            #else
                ct = +15.34824844488746404710*N_half; // TODO: Improve further?
            #endif
        }
        printf("Evaluation yielded %.16f ± %.16f\n", ct, error);

    }
    else{
        // just behavior divergent in N.
        if(this->D==1)  ct = - 2 / N_half;
        if(this->D==2)  ct = 0; // FIX
        if(this->D==3)  ct = +15.34824844488746404710*N_half;
    }

    return ct;
}

//////
//////      DISPERSION DOMAIN
//////

dispersion::dispersion(const unsigned int DD, const unsigned int NN, const double LL, const unsigned int nnstep, bool iimproved):
    cartesian(DD,NN,iimproved)
{

    this->L = LL;
    this->nstep = nnstep;

}

double dispersion::counterterm(const double &x){
    return 0.;  // TODO: Fix, obviously.
}
