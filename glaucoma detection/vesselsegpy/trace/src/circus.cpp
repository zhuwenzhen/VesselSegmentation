//
// Configuration
//

// Header include
#include "circus.hpp"

// Standard library
#include <stdlib.h>  // for abs
#include <new>       // for operator new
#include <algorithm> // for min, swap, max, max_element, etc
#include <cmath>     // for floor
#include <cassert>   // for assert
#include <vector>    // for vector
#include <iostream>  // for cerr, endl

// Eigen
#include <Eigen/SVD> // for JacobiSVD, etc

// Boost
#include <boost/lexical_cast.hpp>    // for lexical_cast, etc
#include <boost/program_options.hpp> // for validation_error, etc

// Local
#include "auxiliary.hpp"
#include "functionals.hpp"


//
// Module definitions
//

std::istream &operator>>(std::istream &in, PFunctionalWrapper &wrapper) {
    in >> wrapper.name;
    if (wrapper.name == "1") {
        wrapper.name = "P1";
        wrapper.functional = PFunctional::P1;
    } else if (wrapper.name == "2") {
        wrapper.name = "P2";
        wrapper.functional = PFunctional::P2;
    } else if (wrapper.name == "3") {
        wrapper.name = "P3";
        wrapper.functional = PFunctional::P3;
    } else if (wrapper.name[0] == 'H') {
        wrapper.functional = PFunctional::Hermite;
        if (wrapper.name.size() < 2) {
            std::cerr << "ERROR: Missing order parameter for Hermite P-functional" << std::endl;
            throw boost::program_options::validation_error(
                boost::program_options::validation_error::invalid_option_value);
        }
        try {
            wrapper.arguments.order =
                boost::lexical_cast<unsigned int>(wrapper.name.substr(1));
        }
        catch (boost::bad_lexical_cast &) {
            std::cerr << "Unparseable order parameter for Hermite P-functional"
                     << std::endl;
            throw boost::program_options::validation_error(
                boost::program_options::validation_error::invalid_option_value);
        }
    } else {
        throw boost::program_options::validation_error(
            boost::program_options::validation_error::invalid_option_value);
    }
    return in;
}

Eigen::MatrixXf nearest_orthonormal_sinogram(const Eigen::MatrixXf &input,
                                             size_t &new_center) {
    // Detect the offset of each column to the sinogram center
    assert(input.rows() > 0 && input.cols() > 0);
    int sinogram_center = std::floor((input.rows() - 1) / 2.0);
    std::vector<int> offset(input.cols()); // TODO: Eigen vector
    for (int p = 0; p < input.cols(); p++) {
        size_t median =
            findWeightedMedian(input.col(p));
        offset[p] = median - sinogram_center;
    }

    // Align each column to the sinogram center
    int min = *(std::min_element(offset.begin(), offset.end()));
    int max = *(std::max_element(offset.begin(), offset.end()));
    assert(sgn(min) != sgn(max));
    int padding = (int)(std::abs(max) + std::abs(min));
    new_center = sinogram_center + max;
    Eigen::MatrixXf aligned =
        Eigen::MatrixXf::Zero(input.rows() + padding, input.cols());
    for (int col = 0; col < input.cols(); col++) {
        for (int row = 0; row < input.rows(); row++) {
            aligned(max + row - offset[col], col) = input(row, col);
        }
    }

    // Compute the nearest orthonormal sinogram
    Eigen::JacobiSVD<Eigen::MatrixXf, Eigen::ColPivHouseholderQRPreconditioner>
    svd(aligned, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXf nos =
        svd.matrixU().block(0, 0, aligned.rows(), aligned.cols()) *
        svd.matrixV().transpose();

    return nos;
}

std::vector<Eigen::VectorXf>
getCircusFunctions(const Eigen::MatrixXf &input,
                   const std::vector<PFunctionalWrapper> &pfunctionals) {
    // Allocate the output matrices
    std::vector<Eigen::VectorXf> outputs(pfunctionals.size());
    for (size_t p = 0; p < pfunctionals.size(); p++)
        outputs[p] = Eigen::VectorXf(input.cols());

    // Pre-calculate
    std::map<size_t, void *> precalculations;
    for (size_t p = 0; p < pfunctionals.size(); p++) {
        PFunctional pfunctional = pfunctionals[p].functional;
        switch (pfunctional) {
        case PFunctional::P3:
            precalculations[p] = PFunctional3_prepare(input.rows());
            break;
        case PFunctional::Hermite:
        case PFunctional::P1:
        case PFunctional::P2:
        default:
            break;
        }
    }

    // Trace all columns
    for (int column = 0; column < input.cols(); column++) {
        Eigen::VectorXf data = input.col(column);

        // Process all P-functionals
        for (size_t p = 0; p < pfunctionals.size(); p++) {
            PFunctional pfunctional = pfunctionals[p].functional;
            float result;
            switch (pfunctional) {
            case PFunctional::P1:
                result = PFunctional1(data);
                break;
            case PFunctional::P2:
                result = PFunctional2(data);
                break;
            case PFunctional::P3:
                result = PFunctional3(data);
                break;
            case PFunctional::Hermite:
                result = PFunctionalHermite(data,
                                            *pfunctionals[p].arguments.order,
                                            *pfunctionals[p].arguments.center);
                break;
            }
            outputs[p](column) = result;
        }
    }

    // Destroy pre-calculations
    std::map<size_t, void *>::iterator it = precalculations.begin();
    while (it != precalculations.end()) {
        PFunctional pfunctional = pfunctionals[it->first].functional;
        switch (pfunctional) {
        case PFunctional::P3: {
            PFunctional3_precalc_t *precalc =
                (PFunctional3_precalc_t *)it->second;
            PFunctional3_destroy(precalc);
            break;
        }
        case PFunctional::Hermite:
        case PFunctional::P1:
        case PFunctional::P2:
        default:
            break;
        }
        ++it;
    }

    return outputs;
}
