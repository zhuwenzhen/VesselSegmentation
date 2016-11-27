//
// Configuration
//

// Header include
#include "transform.hpp"

// Standard library
#include <stddef.h>  // for size_t
#include <algorithm> // for min
#include <cmath>     // for ceil, sqrt
#include <new>       // for operator new
#include <ostream>   // for operator<<, basic_ostream, etc
#include <string>    // for operator<<
#include <vector>    // for vector

// Local
#include "logger.hpp"
#include "auxiliary.hpp"
#include "sinogram.hpp"
#include "circus.hpp"


//
// Module definitions
//

Transformer::Transformer(const Eigen::MatrixXf &image,
                         const std::string &basename,
                         unsigned int angle_stepsize, bool orthonormal)
    : _image(image), _basename(basename), _orthonormal(orthonormal),
      _angle_stepsize(angle_stepsize) {
    // Orthonormal P-functionals need a stretched image in order to ensure a
    // square sinogram
    if (_orthonormal) {
        size_t ndiag = (int)std::ceil(360.0 / angle_stepsize);
        size_t nsize = (int)std::ceil(ndiag / std::sqrt(2));
        clog(debug) << "Stretching input image to " << nsize << " squared."
                    << std::endl;
        _image = resize(_image, nsize, nsize);
    }

    // Pad the images so we can freely rotate without losing information
    _image = pad(_image);
    clog(debug) << "Padded image to " << _image.rows() << "x" << _image.cols()
                << std::endl;
}

void
Transformer::getTransform(const std::vector<TFunctionalWrapper> &tfunctionals,
                          std::vector<PFunctionalWrapper> &pfunctionals,
                          bool write_data) const {
    Eigen::MatrixXf signatures((int)std::floor(360 / _angle_stepsize),
                               tfunctionals.size() * pfunctionals.size());

    // Process all T-functionals
    clog(debug) << "Calculating sinograms for given T-functionals" << std::endl;
    std::vector<Eigen::MatrixXf> sinograms =
        getSinograms(_image, _angle_stepsize, tfunctionals);
    for (size_t t = 0; t < tfunctionals.size(); t++) {
        if (write_data && clog(debug)) {
            // Save the sinogram trace
            std::stringstream fn_trace_data;
            fn_trace_data << _basename << "-" << tfunctionals[t].name << ".csv";
            writecsv(fn_trace_data.str(), sinograms[t]);

            // Save the sinogram image
            std::stringstream fn_trace_image;
            fn_trace_image << _basename << "-" << tfunctionals[t].name
                           << ".pgm";
            writepgm(fn_trace_image.str(), mat2gray(sinograms[t]));
        }

        // Orthonormal functionals require the nearest orthonormal sinogram
        if (_orthonormal) {
            clog(trace) << "Orthonormalizing sinogram" << std::endl;
            size_t sinogram_center;
            sinograms[t] =
                nearest_orthonormal_sinogram(sinograms[t], sinogram_center);
            for (size_t p = 0; p < pfunctionals.size(); p++) {
                if (pfunctionals[p].functional == PFunctional::Hermite) {
                    pfunctionals[p].arguments.center = sinogram_center;
                }
            }
        }

        // Process all P-functionals
        if (pfunctionals.size() > 0) {
            clog(debug) << "Calculating circusfunctions for given P-functionals"
                        << std::endl;
            std::vector<Eigen::VectorXf> circusfunctions =
                getCircusFunctions(sinograms[t], pfunctionals);
            for (size_t p = 0; p < pfunctionals.size(); p++) {
                // Normalize
                Eigen::VectorXf normalized = zscore(circusfunctions[p]);

                if (write_data) {
                    // Aggregate the signatures
                    assert(signatures.rows() == normalized.size());
                    signatures.col(t * pfunctionals.size() + p) = normalized;
                }
            }
        }
    }

    // Save the signatures
    if (write_data && pfunctionals.size() > 0) {
        std::stringstream fn_signatures;
        fn_signatures << _basename << ".csv";
        writecsv(fn_signatures.str(), signatures);
    }
}
