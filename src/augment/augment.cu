#include <stdexcept>

#include "augment.h"

using namespace augment;

template <typename T> __device__ void store(T &, float);

template <> __device__ void store<float>(float &out, float val) { out = val; }

template <> __device__ void store<uint8_t>(uint8_t &out, float val) { out = 255 * __saturatef(val); }

template <typename out_t> struct SurfaceTypeMap;

template <> struct SurfaceTypeMap<uint8_t> { using type = uchar4; };

template <> struct SurfaceTypeMap<float> { using type = float4; };

/**
 * @brief Loads RGB input tensor into a 2D layered texture.
 *
 * @tparam in_t     Input tensor scalar type (uint8_t or float)
 * @tparam surf_t   Corresponding CUDA surface type (uchar4 or float4)
 * @param input     Input pointer
 * @param output    Output 2D layered surface
 * @param width     Input width in pixels
 * @param height    Input height in pixels
 */
template <typename in_t, typename surf_t = typename SurfaceTypeMap<in_t>::type>
__global__ void loadInputTensor(const in_t *input, cudaSurfaceObject_t output, size_t width, size_t height) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    unsigned int i = 3 * ((blockIdx.z * height + y) * width + x);

    const surf_t rgb_{input[i], input[i + 1], input[i + 2], 0};

    surf2DLayeredwrite(rgb_, output, x * sizeof(surf_t), y, blockIdx.z);
}

template <typename out_t>
__global__ void process(cudaTextureObject_t texObj,
                        out_t *out,
                        const size_t width,
                        const size_t height,
                        const size_t batchSize,
                        const Params *params) {
    // get pixel position
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    const auto &imgParams = params[blockIdx.z];

    // calculate normalized texture coordinates
    float u = ((float)x + 0.5f) / width;
    float v = ((float)y + 0.5f) / height;

    // apply translation
    u += imgParams.translation[0];
    v += imgParams.translation[1];

    // apply 3D rotation
    u -= 0.5f;
    v -= 0.5f;
    float z = imgParams.geom[2][2] / (imgParams.geom[2][2] + imgParams.geom[0][2] * u + imgParams.geom[1][2] * v);
    // solving for z of intersection of the outbound ray with the image plane
    float tu = (z * (u * imgParams.geom[0][0] + v * imgParams.geom[1][0]) + (z - 1) * imgParams.geom[2][0]) + 0.5f;
    float tv = (z * (u * imgParams.geom[0][1] + v * imgParams.geom[1][1]) + (z - 1) * imgParams.geom[2][1]) + 0.5f;

    // apply flipping
    if (imgParams.flags & FLAG_HORIZONTAL_FLIP)
        tu = 1.0f - tu;
    if (imgParams.flags & FLAG_VERTICAL_FLIP)
        tv = 1.0f - tv;

    // sample the input texture
    float4 sample = tex2DLayered<float4>(texObj, tu, tv, blockIdx.z);

    // get another sample (Mixup)
    if (blockIdx.z != imgParams.mixImgIdx) {
        if (imgParams.flags & FLAG_MIX_HORIZONTAL_FLIP)
            tu = 1.0f - tu;
        if (imgParams.flags & FLAG_MIX_VERTICAL_FLIP)
            tv = 1.0f - tv;

        float4 sample2 = tex2DLayered<float4>(texObj, tu, tv, imgParams.mixImgIdx);

        sample.x = (1 - imgParams.mixFactor) * sample.x + imgParams.mixFactor * sample2.x;
        sample.y = (1 - imgParams.mixFactor) * sample.y + imgParams.mixFactor * sample2.y;
        sample.z = (1 - imgParams.mixFactor) * sample.z + imgParams.mixFactor * sample2.z;
    }

    // fill surroundings
    if (tu <= 0.0f || tu >= 1.0f || tv <= 0.0f || tv >= 1.0f)
        sample.x = sample.y = sample.z = 0.5f;

    // cutout
    if (imgParams.flags & FLAG_CUTOUT) {
        if (abs(tu - imgParams.cutoutPos[0]) < imgParams.cutoutSize[0] &&
            abs(tv - imgParams.cutoutPos[1]) < imgParams.cutoutSize[1])
            sample.x = sample.y = sample.z = 0.5f;
    }

    // apply color transform
    if (imgParams.gammaCorr != 1) {
        sample.x = __powf(sample.x, imgParams.gammaCorr);
        sample.y = __powf(sample.y, imgParams.gammaCorr);
        sample.z = __powf(sample.z, imgParams.gammaCorr);
    }
    float r = imgParams.color[0][0] * sample.x + imgParams.color[0][1] * sample.y + imgParams.color[0][2] * sample.z;
    float g = imgParams.color[1][0] * sample.x + imgParams.color[1][1] * sample.y + imgParams.color[1][2] * sample.z;
    float b = imgParams.color[2][0] * sample.x + imgParams.color[2][1] * sample.y + imgParams.color[2][2] * sample.z;

    if (imgParams.flags & FLAG_COLOR_INVERSION) {
        r = 1 - r;
        g = 1 - g;
        b = 1 - b;
    }

    // write out
    unsigned int i = 3 * ((blockIdx.z * height + y) * width + x);
    store(out[i], r);
    store(out[i + 1], g);
    store(out[i + 2], b);
}

template <typename in_t, typename out_t>
void compute(cudaStream_t stream,
             const in_t *input,
             out_t *output,
             size_t inWidth,
             size_t inHeight,
             size_t outWidth,
             size_t outHeight,
             size_t batchSize,
             const Params *params) {

    // allocate temporary cuda array
    cudaChannelFormatDesc channelDesc;
    channelDesc.f = std::is_same<in_t, float>::value ? cudaChannelFormatKindFloat : cudaChannelFormatKindUnsigned;
    channelDesc.w = 8 * sizeof(in_t);
    channelDesc.x = 8 * sizeof(in_t);
    channelDesc.y = 8 * sizeof(in_t);
    channelDesc.z = 8 * sizeof(in_t);
    cudaArray *buffer;
    auto error = cudaMalloc3DArray(&buffer,
                                   &channelDesc,
                                   make_cudaExtent(inWidth, inHeight, batchSize),
                                   cudaArrayLayered | cudaArraySurfaceLoadStore);
    if (error != cudaSuccess)
        throw std::runtime_error("Cannot allocate 3D array: " + std::string(cudaGetErrorString(error)));

    // prepare the corresponding resource descriptor
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = buffer;

    // make a surface object for input conversion
    cudaSurfaceObject_t surfObj;
    error = cudaCreateSurfaceObject(&surfObj, &resDesc);
    if (error != cudaSuccess) {
        cudaFreeArray(buffer);
        throw std::runtime_error("Cannot create surface object: " + std::string(cudaGetErrorString(error)));
    }

    // load input tenso to the temporary buffer array
    {
        const dim3 threads(16, 16, 1);
        const dim3 blocks((inWidth + threads.x - 1) / threads.x, (inHeight + threads.y - 1) / threads.y, batchSize);
        loadInputTensor<in_t><<<blocks, threads, 0, stream>>>(input, surfObj, inWidth, inHeight);
    }

    // set up input texture
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = std::is_same<in_t, float>::value ? cudaReadModeElementType : cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;

    cudaTextureObject_t texObj = 0;
    error = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    if (error != cudaSuccess) {
        cudaFreeArray(buffer);
        throw std::runtime_error("Cannot create texture object: " + std::string(cudaGetErrorString(error)));
    }

    // run the main kernel
    const dim3 threads(16, 16, 1);
    const dim3 blocks((outWidth + threads.x - 1) / threads.x, (outHeight + threads.y - 1) / threads.y, batchSize);
    process<<<blocks, threads, 0, stream>>>(texObj, output, outWidth, outHeight, batchSize, params);

    // destroy texture
    cudaDestroyTextureObject(texObj);
    cudaDestroyTextureObject(surfObj);
    cudaFreeArray(buffer);

    // check for errors
    error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}

template <>
void augment::compute(cudaStream_t stream,
                      const uint8_t *input,
                      uint8_t *output,
                      size_t inWidth,
                      size_t inHeight,
                      size_t outWidth,
                      size_t outHeight,
                      size_t batchSize,
                      const Params *params) {
    ::compute(stream, input, output, inWidth, inHeight, outWidth, outHeight, batchSize, params);
}

template <>
void augment::compute(cudaStream_t stream,
                      const uint8_t *input,
                      float *output,
                      size_t inWidth,
                      size_t inHeight,
                      size_t outWidth,
                      size_t outHeight,
                      size_t batchSize,
                      const Params *params) {
    ::compute(stream, input, output, inWidth, inHeight, outWidth, outHeight, batchSize, params);
}

template <>
void augment::compute(cudaStream_t stream,
                      const float *input,
                      float *output,
                      size_t inWidth,
                      size_t inHeight,
                      size_t outWidth,
                      size_t outHeight,
                      size_t batchSize,
                      const Params *params) {
    ::compute(stream, input, output, inWidth, inHeight, outWidth, outHeight, batchSize, params);
}

void augment::setColorTransform(
    Params &params, float hueShiftRad, float saturationFactor, float valueFactor, const float whiteBalanceScales[2]) {
    // Sampling a rotation and scaling matrix in RGB space:
    //   - rotation around (1,1,1) vector by hueShiftRad radians,
    //   - scaling along (1,1,1) vector by valueFactor and in orthogonal
    //   direction by saturationFactor
    static const float sqrt3 = sqrtf(3);
    const float c = cosf(hueShiftRad), s = sinf(hueShiftRad);
    const float _1 = (valueFactor * (12 * saturationFactor * c + 6)) / 18,
                _2 = (valueFactor * (6 * saturationFactor * c + 6 * sqrt3 * saturationFactor * s - 6)) / 18,
                _3 = (valueFactor * (6 * sqrt3 * saturationFactor * s - 6 * saturationFactor * c + 6)) / 18;

    params.color[0][0] = _1;
    params.color[0][1] = -_2;
    params.color[0][2] = _3;

    params.color[1][0] = _3;
    params.color[1][1] = _1;
    params.color[1][2] = -(valueFactor * (saturationFactor * c + sqrt3 * saturationFactor * s - 1)) / 3;

    params.color[2][0] = -_2;
    params.color[2][1] = _3;
    params.color[2][2] = (valueFactor * (4 * saturationFactor * c + 2)) / 6;

    // pre-multiply by white balance scales
    params.color[0][0] *= whiteBalanceScales[0];
    params.color[1][0] *= whiteBalanceScales[0];
    params.color[2][0] *= whiteBalanceScales[0];

    params.color[0][2] *= whiteBalanceScales[0];
    params.color[1][2] *= whiteBalanceScales[0];
    params.color[2][2] *= whiteBalanceScales[0];
}

void augment::setGeometricTransform(Params &params, float pan, float tilt, float roll, float scaleX, float scaleY) {
    /*
        X, Y: image axes, Z: forward

        Rotation in XZ plane (tilt):
            [ cos(a)  0  sin(a)
                   0  1       0
             -sin(a)  0  cos(a) ]

        Rotation in YZ plane (pan):
            [ 1        0      0
              0   cos(b) sin(b)
              0  -sin(b) cos(b) ]

        Rotation in XY plane (roll):
            [  cos(c) sin(c)  0
              -sin(c) cos(c)  0
                    0      0  1 ]

        Considering the image is on Z=0 plane, the camera is at Z=-1 point

        full transform R = XY * YZ * XZ * Diag(scaleX, scaleY, 1)
    */

    const float cosA = std::cos(pan), sinA = std::sin(pan), cosB = std::cos(tilt), sinB = std::sin(tilt),
                cosC = std::cos(roll), sinC = std::sin(roll);

    params.geom[0][0] = -sinA * sinB * sinC + cosA * cosC;
    params.geom[0][1] = sinC * cosB;
    params.geom[0][2] = sinA * cosC + sinB * sinC * cosA;

    params.geom[1][0] = -sinA * sinB * cosC - sinC * cosA;
    params.geom[1][1] = cosB * cosC;
    params.geom[1][2] = -sinA * sinC + sinB * cosA * cosC;

    params.geom[2][0] = -sinA * cosB;
    params.geom[2][1] = -sinB;
    params.geom[2][2] = cosA * cosB;

    // apply scaling factors
    params.geom[0][0] *= scaleX;
    params.geom[0][1] *= scaleX;
    params.geom[0][2] *= scaleX;

    params.geom[1][0] *= scaleY;
    params.geom[1][1] *= scaleY;
    params.geom[1][2] *= scaleY;
}