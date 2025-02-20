#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace augment {

static const unsigned int FLAG_HORIZONTAL_FLIP = 1, FLAG_VERTICAL_FLIP = 2, FLAG_MIX_HORIZONTAL_FLIP = 4,
                          FLAG_MIX_VERTICAL_FLIP = 8, FLAG_CUTOUT = 16, FLAG_COLOR_INVERSION = 32;

/**
 * Transformations applied to a single image in a batch
 */
typedef struct {
    float color[3][3];    //!< matrix applied in RGB color space
    float gammaCorr;      //!< gamma correction factor
    float geom[3][3];     //!< geometrical transformation
    float translation[2]; //!< normalized translation
    float cutoutPos[2];
    float cutoutSize[2];
    unsigned int mixImgIdx; //!< index of a second image in the batch to mix to
                            //!< the current image
    float mixFactor;        //!< weight of the second image in the mix (0 = no mix)
    unsigned int flags;
} Params;

void setColorTransform(
    Params &params, float hueShiftRad, float saturationFactor, float valueFactor, const float whiteBalanceScales[2]);

void setGeometricTransform(Params &params, float pan, float tilt, float roll, float scaleX, float scaleY);

template <typename in_t, typename out_t>
void compute(cudaStream_t stream,
             const in_t *input,
             out_t *output,
             size_t inWidth,
             size_t inHeight,
             size_t outWidth,
             size_t outHeight,
             size_t batchSize,
             const Params *params);

} // namespace augment