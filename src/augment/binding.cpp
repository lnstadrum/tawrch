#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "kernel_base.hpp"

typedef std::variant<float, std::vector<float>> ScalarOrList;

void getPair(const ScalarOrList &input, const char *attrName, float &x, float &y) {
    if (std::holds_alternative<float>(input)) {
        x = y = std::get<float>(input);
        return;
    }

    const auto &vec = std::get<std::vector<float>>(input);
    const auto size = vec.size();
    if (size == 0 || size > 2)
        throw std::invalid_argument("Expecting one or two entries in '" + std::string(attrName) +
                                    "' argument, but got " + std::to_string(size) + ".");

    if (size == 1)
        x = y = vec[0];
    else {
        x = vec[0];
        y = vec[1];
    }
}

/**
 * Temporary GPU memory buffer implementation using CUDACachingAllocator
 * The cache needs to be cleaned.
 */
class TorchTempGPUBuffer {
  private:
    c10::DataPtr ptr;
    c10::cuda::CUDAStream stream;

  public:
    TorchTempGPUBuffer(size_t sizeBytes, c10::cuda::CUDAStream stream)
        : ptr(c10::cuda::CUDACachingAllocator::get()->allocate(sizeBytes)), stream(stream) {}

    ~TorchTempGPUBuffer() { c10::cuda::CUDACachingAllocator::recordStream(ptr, stream); }

    inline uint8_t *operator()() { return reinterpret_cast<uint8_t *>(ptr.get()); }
};

/**
 * Data augmentation kernel for PyTorch
 */
class Kernel : public augment::KernelBase<TorchTempGPUBuffer, c10::cuda::CUDAStream> {
  private:
    using Base = augment::KernelBase<TorchTempGPUBuffer, c10::cuda::CUDAStream>;

    augment::Settings settings;

    template <typename... Args>
    inline void launchKernel(const torch::Tensor &input, torch::Tensor &output, Args... args) {
        if (input.scalar_type() == torch::kUInt8) {
            if (output.scalar_type() == torch::kUInt8) {
                Base::run(settings, input.data_ptr<uint8_t>(), output.data_ptr<uint8_t>(), args...);
                return;
            }
            if (output.scalar_type() == torch::kFloat) {
                Base::run(settings, input.data_ptr<uint8_t>(), output.data_ptr<float>(), args...);
                return;
            }
        } else if (input.scalar_type() == torch::kFloat) {
            if (output.scalar_type() == torch::kFloat) {
                Base::run(settings, input.data_ptr<float>(), output.data_ptr<float>(), args...);
                return;
            }
        }

        throw std::runtime_error("Unsupported input/output datatype combination");
    }

  public:
    Kernel(int seed,
           const ScalarOrList &translation,
           const ScalarOrList &scale,
           float prescale,
           float rotation,
           const ScalarOrList &perspective,
           float cutout,
           const ScalarOrList &cutoutSize,
           float mixup,
           float mixupAlpha,
           float hue,
           float saturation,
           float brightness,
           float whiteBalance,
           float gammaCorrection,
           bool colorInversion,
           bool flipHorizontally,
           bool flipVertically) {
        // get translation range
        getPair(translation, "translation", settings.translation[0], settings.translation[1]);

        // get scaling parameters
        getPair(scale, "scale", settings.scale[0], settings.scale[1]);
        settings.isotropicScaling =
            !std::holds_alternative<std::vector<float>>(scale) || std::get<std::vector<float>>(scale).size() == 1;
        settings.prescale = prescale;

        // get rotation in radians
        settings.rotation = rotation * pi / 180.0f;

        // get perspective angles and convert to radians
        getPair(perspective, "perspective", settings.perspective[0], settings.perspective[1]);
        settings.perspective[0] *= pi / 180.0f;
        settings.perspective[1] *= pi / 180.0f;

        // get flipping flags
        settings.flipHorizontally = flipHorizontally;
        settings.flipVertically = flipVertically;

        // get CutOut parameters
        settings.cutoutProb = cutout;
        settings.cutout[0] = settings.cutout[1] = 0.0f;
        if (std::holds_alternative<std::vector<float>>(cutoutSize) && std::get<std::vector<float>>(cutoutSize).empty())
            settings.cutoutProb = 0;
        else
            getPair(cutoutSize, "cutout_size", settings.cutout[0], settings.cutout[1]);

        // get Mixup parameters
        settings.mixupProb = mixup;
        settings.mixupAlpha = mixupAlpha;

        // get color correction parameters
        settings.hue = hue * pi / 180.0f;
        settings.saturation = saturation;
        settings.brightness = brightness;
        settings.whiteBalance = whiteBalance;
        settings.colorInversion = colorInversion;
        settings.gammaCorrection = gammaCorrection;

        // set random seed
        if (seed != 0)
            augment::KernelBase<TorchTempGPUBuffer, c10::cuda::CUDAStream>::setRandomSeed(seed);

        // check settings
        settings.check();
    }

    Kernel() { settings.setBypass(); }

    std::variant<std::vector<torch::Tensor>, torch::Tensor>
    operator()(const torch::Tensor &input,
               const std::optional<torch::Tensor> &labels,
               const std::optional<std::vector<int64_t>> &outputSize,
               torch::Dtype outputType,
               bool outputMapping) {
        // check output size
        if (outputSize && !outputSize->empty() && outputSize->size() != 2)
            throw std::invalid_argument("Invalid output_size: expected 2 entries, got " +
                                        std::to_string(outputSize->size()));

        // check the input tensor
        if (input.dim() < 3 || input.dim() > 5)
            throw std::invalid_argument("Expected a 3-, 4- or 5-dimensional input tensor, got " +
                                        std::to_string(input.dim()) + " dimensions");
        if (!input.is_cuda())
            throw std::invalid_argument("Expected an input tensor in GPU memory (likely missing a .cuda() "
                                        "call)");
        if (!input.is_contiguous())
            throw std::invalid_argument("Expected a contiguous input tensor");
        if (input.scalar_type() != torch::kUInt8 && input.scalar_type() != torch::kFloat)
            throw std::invalid_argument("Expected uint8 or float input tensor");

        // get input sizes
        const int64_t batchSize = input.dim() >= 4 ? input.size(0) : 0, groups = input.dim() == 5 ? input.size(1) : 0,
                      inputHeight = input.size(input.dim() - 3), inputWidth = input.size(input.dim() - 2),
                      inputChannels = input.size(input.dim() - 1),
                      outputWidth = !outputSize || outputSize->empty() ? inputWidth : (*outputSize)[0],
                      outputHeight = !outputSize || outputSize->empty() ? inputHeight : (*outputSize)[1];

        // check number of input channels
        if (inputChannels != 3)
            throw std::invalid_argument("Expected a 3-channel channels-last (*HW3) input tensor, got " +
                                        std::to_string(inputChannels) + " channels");

        // get CUDA stream
        auto stream = c10::cuda::getCurrentCUDAStream(input.device().index());

        // get input labels tensor
        const auto numClasses = labels ? labels->size(labels->dim() - 1) : 0;
        if (labels) {
            const auto expectedDims = input.dim() - 2;
            if (labels->dim() != expectedDims)
                throw std::invalid_argument("Expected a " + std::to_string(expectedDims) +
                                            "-dimensional input_labels tensor, got " + std::to_string(labels->dim()) +
                                            " dimensions");
            if (labels->size(0) != batchSize)
                throw std::invalid_argument("First dimension of the input labels tensor is expected to match "
                                            "the batch size, but got " +
                                            std::to_string(labels->size(0)));

            const auto expectedNumElems = batchSize * std::max<int64_t>(groups, 1) * numClasses;
            if (labels->numel() != expectedNumElems)
                throw std::invalid_argument("Expected " + std::to_string(expectedNumElems) +
                                            " elements in the labels tensor but got " +
                                            std::to_string(labels->numel()));

            if (!labels->is_cpu())
                throw std::invalid_argument("Expected an input_labels tensor stored in RAM (likely missing a "
                                            ".cpu() call)");
            if (labels->scalar_type() != torch::kFloat)
                throw std::invalid_argument("Expected a floating-point input_labels tensor");
        }
        const float *inputLabelsPtr = labels ? labels->expect_contiguous()->data_ptr<float>() : nullptr;

        // allocate output tensors
        auto outputOptions = torch::TensorOptions().device(input.device()).dtype(outputType);
        std::vector<int64_t> outputShape{outputHeight, outputWidth, 3};
        if (groups > 0)
            outputShape.emplace(outputShape.begin(), groups);
        if (batchSize > 0)
            outputShape.emplace(outputShape.begin(), batchSize);

        torch::Tensor output = torch::empty(outputShape, outputOptions);

        torch::Tensor outputLabels;
        if (labels)
            outputLabels = torch::empty_like(*labels);

        torch::Tensor mapping;
        if (outputMapping) {
            std::vector<int64_t> shape{3, 3};
            if (groups > 0)
                shape.emplace(shape.begin(), groups);
            if (batchSize > 0)
                shape.emplace(shape.begin(), batchSize);
            auto opts = torch::TensorOptions().dtype(torch::kFloat32);
            mapping = torch::empty(shape, opts);
        }
        auto outputMappingPtr = outputMapping ? mapping.expect_contiguous()->data_ptr<float>() : nullptr;

        // launch the kernel
        launchKernel(input,
                     output,
                     inputLabelsPtr,
                     labels ? outputLabels.data_ptr<float>() : nullptr,
                     outputMappingPtr,
                     std::max<int64_t>(batchSize, 1),
                     std::max<int64_t>(groups, 1),
                     inputHeight,
                     inputWidth,
                     outputHeight,
                     outputWidth,
                     numClasses,
                     stream.stream(),
                     stream);

        if (labels && outputMapping)
            return std::vector<torch::Tensor>{output, outputLabels, mapping};
        if (labels)
            return std::vector<torch::Tensor>{output, outputLabels};
        if (outputMapping)
            return std::vector<torch::Tensor>{output, mapping};
        else
            return output;
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    py::class_<Kernel>(module, "Augment")
        .def(py::init<>(),
             "Creates an Augment instance which does not do any augmentation and acts as identity transformation.")

        .def(py::init<int,
                      const ScalarOrList &,
                      const ScalarOrList &,
                      float,
                      float,
                      const ScalarOrList &,
                      float,
                      const ScalarOrList &,
                      float,
                      float,
                      float,
                      float,
                      float,
                      float,
                      float,
                      bool,
                      bool,
                      bool>(),
             py::arg("seed"),
             py::arg("translation") = 0.1f,
             py::arg("scale") = std::vector<float>{0.1},
             py::arg("prescale") = 1.0f,
             py::arg("rotation") = 15.0f,
             py::arg("perspective") = 15.0f,
             py::arg("cutout") = 0.5f,
             py::arg("cutout_size") = std::vector<float>{0.3f, 0.5f},
             py::arg("mixup") = 0.0f,
             py::arg("mixup_alpha") = 0.4f,
             py::arg("hue") = 10.0f,
             py::arg("saturation") = 0.4f,
             py::arg("brightness") = 0.1f,
             py::arg("white_balance") = 0.5f,
             py::arg("gamma_corr") = 0.2f,
             py::arg("color_inversion") = false,
             py::arg("flip_horizontally") = true,
             py::arg("flip_vertically") = false,
             R"""(Creates a FastAugment object used to apply a set of random geometry and
             color transformations to batches of images.

             Args:
                 translation:        Normalized image translation range along X and Y axis.
                                     `0.1` corresponds to a random shift by at most 10% of
                                     the image size in both directions (default).
                                     If one value given, the same range applies for X and Y
                                     axes.
                 scale:              Scaling factor range along X and Y axes. `0.1`
                                     corresponds to stretching the images by a random factor
                                     of at most 10% (default).
                                     If one value given, the applied scaling keeps the
                                     aspect ratio: the same factor is used along X and Y
                                     axes.
                 prescale:           A constant scaling factor applied to all images. Can be
                                     used to shift the random scaling distribution from its
                                     default average equal to 1 and crop out image borders.
                                     Higher values make the output images appear smaller.
                                     The default value is 1.
                 rotation:           Rotation angle range in degrees. The images are rotated
                                     in both clockwise and counter-clockwise direction by a
                                     random angle less than `rotation`. Default: 10 degrees.
                 perspective:        Perspective distortion range setting the maximum
                                     tilting and panning angles in degrees.
                                     The image plane is rotated in 3D around X and Y axes
                                     (tilt and pan respectively) by random angles smaller
                                     than the given value(s).
                                     If one number is given, the same range applies for both
                                     axes. The default value is 15 degrees.
                 flip_horizontally:  A boolean. If `True`, the images are flipped
                                     horizontally with 50% chance. Default: True.
                 flip_vertically:    A boolean. If `True`, the images are flipped vertically
                                     with 50% chance. Default: False.
                 hue:                Hue shift range in degrees. The image pixels color hues
                                     are shifted by a random angle smaller than `hue`.
                                     A hue shift of +/-120 degrees transforms green in
                                     red/blue and vice versa. The default value is 10 deg.
                 saturation:         Color saturation factor range. For every input image,
                                     the color saturation is scaled by a random factor
                                     sampled in range `[1 - saturation, 1 + saturation]`.
                                     Applying zero saturation scale produces a grayscale
                                     image. The default value is 0.4.
                 brightness:         Brightness factor range. For every input image, the
                                     intensity is scaled by a random factor sampled in range
                                     `[1 - brightness, 1 + brightness]`.
                                     The default value is 0.1.
                 white_balance:      White balance scale range in number of stops.
                                     Random gains are applied to red and blue channels.
                                     The default value is 0.5.
                 gamma_corr:         Gamma correction factor range. For every input image,
                                     the factor value is randomly sampled in range
                                     `[1 - gamma_corr, 1 + gamma_corr]`.
                                     Gamma correction boosts (for factors below 1) or
                                     reduces (for factors above 1) dark image areas
                                     intensity, while bright areas are less affected.
                                     The default value is 0.2.
                 color_inversion:    A boolean. If `True`, colors of all pixels in every
                                     image are inverted (negated) with 50% chance.
                                     Default: False.
                 cutout:             Probability of CutOut being applied to a given input
                                     image. The default value is 0.5.
                                     CutOut erases a randomly placed rectangular area of an
                                     image. See the original paper for more details:
                                     https://arxiv.org/pdf/1708.04552.pdf
                 cutout_size:        A list specifying the normalized size range CutOut area
                                     width and height are sampled from.
                                     `[0.3, 0.5]` range produces a rectangle of 30% to 50%
                                     of image size on every side (default).
                                     If an empty list is passed, CutOut application is
                                     disabled.
                 mixup:              Probability of mixup being applied to a given input
                                     image. Mixup is disabled by default (`mixup` is set to
                                     zero).
                                     Mixup is applied across the batch. Every two mixed
                                     images undergo the same set of other transformations
                                     except flipping which can be different.
                                     Requires the input labels `y`. If not provided, an
                                     exception is thrown.
                 mixup_alpha:        Mixup `alpha` parameter (default: 0.4). See the
                                     original paper for more details:
                                     https://arxiv.org/pdf/1710.09412.pdf
                 seed:               Random seed. If different from 0, reproduces the same
                                     sequence of transformations for a given set of
                                     parameters and input size.

             Returns:
                 A `Tensor` with a set of transformations applied to the input image or
                 batch, and another `Tensor` containing the image labels in one-hot format.
             )""")

        .def("set_seed",
             &Kernel::setRandomSeed,
             py::arg("seed"),
             R"""(Reinitializes state of the internal random generator according to a given seed.

             Args:
                 seed (int): the seed value
             )""")

        .def(
            "__call__",
            [](Kernel &kernel,
               const torch::Tensor &input,
               const std::optional<torch::Tensor> &labels,
               const std::optional<std::vector<int64_t>> &outputSize,
               py::object outputType,
               bool outputMapping) {
                return kernel(input,
                              labels,
                              outputSize,
                              py::none().is(outputType) ? input.scalar_type()
                                                        : torch::python::detail::py_object_to_dtype(outputType),
                              outputMapping);
            },
            py::arg("input"),
            py::arg("input_labels") = py::none(),
            py::arg("output_size") = py::none(),
            py::arg("output_type") = py::none(),
            py::arg("output_mapping") = false,
            R"""(Applies a sequence of random transformations to images in a batch.

             Args:
                 x:                  A `Tensor` of `uint8` or `float32` type containing an input
                                     image or batch in channels-last layout (`HWC` or `NHWC`).
                                     3-channel color images are expected (`C=3`).
                 y:                  A `Tensor` of `float32` type containing input labels in
                                     one-hot format. Its outermost dimension is expected to match
                                     the batch size. Optional, can be empty or None.
                 output_size:        A list `[W, H]` specifying the output batch width and height
                                     in pixels. If none, the input size is kept (default).
                 output_type:        Output image datatype. Can be `float32` or `uint8`.
                                     Default: `float32`.
                 output_mapping:     If `True`, the applied transformations are given as the
                                     last output argument. These are 3x3 matrices mapping input
                                     homogeneous coordinates in pixels to output coordinates in
                                     pixels.
             )""");
}