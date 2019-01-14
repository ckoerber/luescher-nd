#include "zeta.hpp"

inline double sum_zeta2(const double xi, const ivec &sum_range){
    return std::accumulate(
        sum_range.begin(),
        sum_range.end(),
        0.0,
        [&](double acc, const int n){
            return acc + 1./(n*n - xi);
    });
    return 0.;
}

dvec zeta2(const dvec &x, const long lambda){
    const double offset(-2*M_PI*std::log(lambda));
    dvec res(x.size(), offset);

    const ivec sum_range([&](){
        ivec vec;
        for(int n = -lambda; n < lambda; ++n){
            vec.push_back(n);
        }
        return vec;
    }());

    std::transform(
        x.begin(),
        x.end(),
        res.begin(),
        [&](const double xi){
            return sum_zeta2(xi, sum_range);
        }
    );

    return res;
}


void test(){
    dvec x(10, 2.0);

    std::cout<< zeta2(x, 20)[0] << std::endl;
}

int main(){

    test();

    return 0;
}
