#include "zeta.hpp"

inline double sum_zeta2(const double xi, const ivec &sum_range){
    const double one(1);
    return std::accumulate(
        sum_range.begin(),
        sum_range.end(),
        0.0,
        [&](double acc, const int n1){
            const double n1sq = n1*n1;
            const double n1sq_m_xi = n1sq - xi;
            return acc + std::accumulate(
                sum_range.begin(),
                sum_range.end(),
                0.0,
                [&](double acc2, const int n2){
                    const double n2sq = n2*n2;
                    return acc2 + one/(n1sq_m_xi + n2sq);
                }
            );
    });
}

dvec zeta2(const dvec &x, const long lambda){
    const double offset(2*M_PI*std::log(lambda));
    dvec res(x.size(), 0);

    const ivec sum_range([&](){
        ivec vec;
        for(long n = -lambda; n <= lambda; ++n){
            vec.push_back(n);
        }
        return vec;
    }());

    std::transform(
        x.begin(),
        x.end(),
        res.begin(),
        [&](const double xi){
            return sum_zeta2(xi, sum_range) - offset;
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
