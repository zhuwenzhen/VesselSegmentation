//
// Configuration
//

// Include guard
#ifndef _TRACETRANSFORM_CIRCUS_
#define _TRACETRANSFORM_CIRCUS_

// Standard library
#include <cstddef> // for size_t
#include <iosfwd>  // for istream
#include <string>  // for string
#include <vector>

// Boost
#include <boost/none.hpp>     // for none
#include <boost/optional.hpp> // for optional

// Eigen
#include <Eigen/Dense> // for MatrixXf, VectorXf


//
// Functionals
//

enum class PFunctional {
    Hermite,
    P1,
    P2,
    P3
};

struct PFunctionalArguments {
    PFunctionalArguments(boost::optional<unsigned int> _order = boost::none,
                         boost::optional<size_t> _center = boost::none)
        : order(_order), center(_center) {}

    // Arguments for Hermite P-functional
    boost::optional<unsigned int> order;
    boost::optional<size_t> center;
};

struct PFunctionalWrapper {
    PFunctionalWrapper() : name("invalid"), functional(PFunctional()) {
        // Invalid constructor, only used by boost::program_options
    }

    PFunctionalWrapper(const std::string &_name, const PFunctional &_functional,
                       const PFunctionalArguments &_arguments =
                           PFunctionalArguments())
        : name(_name), functional(_functional), arguments(_arguments) {}

    std::string name;
    PFunctional functional;
    PFunctionalArguments arguments;
};

std::istream &operator>>(std::istream &in, PFunctionalWrapper &wrapper);


//
// Module definitions
//

Eigen::MatrixXf nearest_orthonormal_sinogram(const Eigen::MatrixXf &input,
                                             size_t &new_center);

std::vector<Eigen::VectorXf>
getCircusFunctions(const Eigen::MatrixXf &input,
                   const std::vector<PFunctionalWrapper> &pfunctionals);

#endif
