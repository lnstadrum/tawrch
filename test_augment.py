from tawrch import Augment
import numpy
import torch
import pytest


@pytest.fixture
def bypass_params():
    # parameters disabling all augmentation transformations (for debugging purposes)
    return {
        "seed": 0,
        "translation": 0,
        "scale": 0,
        "prescale": 1,
        "rotation": 0,
        "perspective": 0,
        "flip_horizontally": False,
        "flip_vertically": False,
        "cutout": 0,
        "cutout_size": 0,
        "mixup": 0,
        "saturation": 0,
        "brightness": 0,
        "hue": 0,
        "gamma_corr": 0,
        "color_inversion": False,
    }


class TestShape:
    """Output shape verification"""

    def test_default_output_size(self):
        input_batch = torch.zeros(5, 123, 234, 3, dtype=torch.uint8).cuda()
        output_batch = Augment(123)(input_batch)
        assert output_batch.shape == input_batch.shape

    def test_specific_output_size(self):
        input_batch = torch.zeros(7, 55, 66, 3, dtype=torch.uint8).cuda()
        width, height = 88, 77
        output_batch = Augment(123)(input_batch, output_size=[width, height])
        assert output_batch.shape == (7, height, width, 3)

    def test_input_dims(self):
        augment = Augment(123)
        x = augment(torch.zeros(10, 20, 3).cuda())
        x.shape == (10, 20, 3)
        x = augment(torch.zeros(5, 10, 20, 3).cuda())
        x.shape == (5, 10, 20, 3)
        x = augment(torch.zeros(5, 2, 10, 20, 3).cuda())
        x.shape == (5, 2, 10, 20, 3)


class TestColors:
    """Color-related tests"""

    def test_identity_u8(self):
        # make random input
        input_batch = torch.randint(size=(5, 23, 45, 3), high=255)
        input_batch = input_batch.to(torch.uint8).cuda()

        # apply identity transformation
        augment = Augment()
        output_batch = augment(input_batch, output_type=torch.uint8)

        # compare: expected same output
        assert torch.equal(output_batch, input_batch)

    def test_identity_u32(self):
        # make random input
        input_batch = torch.randint(size=(5, 23, 45, 3), high=1).to(torch.float32)
        input_batch = input_batch.cuda()

        # apply identity transformation
        augment = Augment()
        output_batch = augment(input_batch, output_type=torch.float32)

        # compare: expected same output
        assert torch.equal(output_batch, input_batch)

    def test_center_pixel(self, bypass_params):
        # make random grayscale input
        input_batch = torch.randint(size=(32, 17, 17, 1), high=255)
        input_batch = input_batch.to(torch.uint8).cuda()
        input_batch = input_batch.repeat(1, 1, 1, 3)

        # apply transformations keeping the center pixel color unchanged
        bypass_params["flip_vertically"] = True
        bypass_params["flip_horizontally"] = True
        bypass_params["perspective"] = [10, 20]
        augment = Augment(**bypass_params)
        output_batch = augment(input_batch, output_type=torch.uint8)

        # compare center pixel colors: expecting the same
        assert torch.equal(output_batch[:, 8, 8, :], input_batch[:, 8, 8, :])

    def test_color_inversion_u8(self, bypass_params):
        # make random input
        input_batch = torch.zeros(5, 23, 45, 3).to(torch.uint8).cuda()

        # apply color inversion only
        bypass_params["color_inversion"] = True
        output_batch = Augment(**bypass_params)(input_batch, output_type=torch.uint8)

        # compare colors
        input_batch = input_batch.cpu().numpy()
        output_batch = output_batch.cpu().numpy()
        comp = numpy.logical_xor(
            input_batch == output_batch, input_batch == 255 - output_batch
        )
        assert numpy.all(comp)

    def test_color_inversion_f32(self, bypass_params):
        # make random input
        input_batch = torch.rand(5, 23, 45, 3).cuda()

        # apply color inversion only
        bypass_params["color_inversion"] = True
        output_batch = Augment(**bypass_params)(input_batch, output_type=torch.float32)

        # compare colors
        diff1 = input_batch - output_batch
        diff2 = 1 - input_batch - output_batch
        assert torch.allclose(diff1 * diff2, torch.zeros_like(diff1))


class TestMixup:
    """Tests of labels computation with mixup"""

    def test_no_mixup(self):
        # make random input
        input_batch = torch.randint(size=(8, 5, 8, 8, 3), high=255)
        input_batch = input_batch.to(torch.uint8).cuda()
        input_labels = torch.rand(size=(8, 5, 1000))

        # apply random transformation
        _, output_labels = Augment(123)(input_batch, input_labels)

        # compare labels: expected same
        assert torch.equal(input_labels, output_labels)

    def test_yes_mixup(self):
        # make random labels
        input_labels = torch.randint(size=(50,), high=2, dtype=torch.int32)

        # make images from labels
        input_batch = (255 * input_labels).to(torch.uint8).cuda()
        input_batch = input_batch.reshape(-1, 1, 1, 1).repeat(1, 5, 5, 3)

        # transform labels to one-hot
        input_labels = torch.nn.functional.one_hot(input_labels.to(torch.long), 2) \
            .to(torch.float32)

        # apply mixup
        augment = Augment(
            seed=0,
            rotation=0,
            flip_vertically=True,
            flip_horizontally=True,
            hue=0,
            saturation=0,
            brightness=0,
            gamma_corr=0,
            cutout=0,
            mixup=0.9,
        )
        output_batch, output_labels = augment(input_batch, input_labels, output_type=torch.float32)

        # check that probabilities sum up to 1
        assert torch.allclose(output_labels[:, 0] + output_labels[:, 1], torch.ones((50)))

        # compare probabilities to center pixel values
        assert torch.allclose(output_labels[:, 1], output_batch[:, 3, 3, 0].cpu())


class TestSeed:
    def test_seed(self):
        # make random input
        input_batch = torch.randint(size=(16, 50, 50, 3), high=255)
        input_batch = input_batch.to(torch.uint8).cuda()

        # make random labels
        input_labels = torch.randint(size=(16,), high=2).to(torch.long)
        input_proba = torch.nn.functional.one_hot(input_labels, 20).to(torch.float32)

        # generate output batches
        output_batch1, output_proba1 = Augment(mixup=0.75, seed=123)(
            input_batch, input_proba
        )
        output_batch2, output_proba2 = Augment(mixup=0.75, seed=234)(
            input_batch, input_proba
        )
        output_batch3, output_proba3 = Augment(mixup=0.75, seed=123)(
            input_batch, input_proba
        )

        # compare
        assert not torch.equal(output_batch1, output_batch2)
        assert not torch.equal(output_proba1, output_proba2)
        assert torch.equal(output_batch1, output_batch3)
        assert torch.equal(output_proba1, output_proba3)

    def test_seed_reset(self):
        # make random input
        input_batch = torch.randint(size=(16, 50, 50, 3), high=255)
        input_batch = input_batch.to(torch.uint8).cuda()

        # make random labels
        input_labels = torch.randint(size=(16,), high=2).to(torch.long)
        input_proba = torch.nn.functional.one_hot(input_labels, 20).to(torch.float32)

        augment = Augment(mixup=0.75, seed=123)

        # generate output batches
        output_batch1, output_proba1 = augment(input_batch, input_proba)
        output_batch2, output_proba2 = augment(input_batch, input_proba)

        # set back the initial seed and generate another output batch
        augment.set_seed(123)
        output_batch3, output_proba3 = augment(input_batch, input_proba)

        # compare
        assert not torch.equal(output_batch1, output_batch2)
        assert not torch.equal(output_proba1, output_proba2)
        assert torch.equal(output_batch1, output_batch3)
        assert torch.equal(output_proba1, output_proba3)


class TestDatatype:
    """Datatype verification"""

    def test_uint8_vs_float32(self):
        # make random input
        input_batch = torch.randint(size=(64, 32, 32, 3), high=255)
        input_batch = input_batch.to(torch.uint8).cuda()

        # apply identical transformation
        augment = Augment(seed=96)
        output_batch_ref = augment(input_batch, output_type=torch.uint8)
        augment.set_seed(96)
        output_batch_float = augment(input_batch, output_type=torch.float32)

        # check output types
        assert output_batch_ref.dtype == torch.uint8
        assert output_batch_float.dtype == torch.float32

        # cast back to uint8 and compare: expecting the same output
        output_batch_test = (255 * output_batch_float.clamp(0, 1)).to(torch.uint8)
        assert torch.equal(output_batch_ref, output_batch_test)


class TestCoordinatesMapping:
    def test_coordinates_mapping(self):
        # generate random batch of zeros with a bright spot at a known position
        input_batch = torch.zeros((30, 2, 120, 250, 3), dtype=torch.uint8).cuda()
        y, x = 28, 222
        input_batch[..., y-2:y+2, x-2:x+2, :] = 255

        # perform augmentation
        augment = Augment(seed=123,
                          gamma_corr=0,
                          brightness=0,
                          hue=0,
                          saturation=0,
                          mixup=0,
                          cutout=0,
                          translation=0.1,
                          rotation=30,
                          scale=0.2,
                          perspective=15,
                          flip_horizontally=True,
                          flip_vertically=True,
                          prescale=2.0)
        output_batch, mappings = augment(input_batch,
                                         output_type=torch.uint8,
                                         output_mapping=True,
                                         output_size=(400, 400))

        # get coordinates of the spot in the augmented images
        coords = torch.matmul(mappings, torch.tensor([x, y, 1], dtype=torch.float32).t())
        coords = (coords[..., :2] / coords[..., 2:3]).round().to(torch.int32).numpy()

        # make sure it is in the output images
        for group in zip(output_batch, coords):
            for image, (x, y) in zip(*group):
                if x >= 0 and x < image.shape[-2] and y >= 0 and y < image.shape[-3]:
                    assert image[y, x, 0] == 255
