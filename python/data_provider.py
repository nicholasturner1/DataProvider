#!/usr/bin/env python
__doc__ = """

DaraProvider classes.

Kisuk Lee <kisuklee@mit.edu>, 2015-2016
"""

from collections import OrderedDict
import numpy as np
import parser
from dataset import VolumeDataset, SampleDataset
from data_augmentation import DataAugmentor
from transform import *
from label_transform import *

class DataProvider(object):
    """
    DataProvider interface.
    """

    def next_sample(self):
        raise NotImplementedError

    def random_sample(self):
        raise NotImplementedError


class BatchDataProvider(DataProvider):
    """
    DataProvider for samples.

    Attributes:
        datasets: List of datasets.
    """

    def __init__(self, dsets, layer_types, net_spec, preps, xforms, augs):
        """
        Initialize DataProvider.

        Args:
            dspec_path: Path to the dataset specification file.
            net_spec:   Net specification.
            params:     Various options.
            auto_mask:  Whether to automatically generate mask from
                        corresponding label.
        """

        self.datasets = list()
        for d in dsets:
            dataset = SampleDataset(d, layer_types, net_spec, preps, xforms)
            self.datasets.append(dataset)
            
        self._data_aug = DataAugmentor(augs)

    def next_sample(self):
        """Fetch next sample in a sample sequence."""
        return self.random_sample()  # TODO(kisuk): Temporary.

    def random_sample(self):
        """Fetch a batch of randomly augmented samples."""

        for dset in self.datasets:
            # Draw a random sample and apply data augmenation.
            sample, transform = self._data_aug.random_sample(dset)
            # Transform sample.
            sample = self._transform(sample, transform)

            samples.append(OrderedDict(sorted(sample.items(), key=lambda k: k)))

        #merging samples to form a batch
        self._make_batch(samples)


    def _make_batch(self, samples):

        assert(len(samples) >= 1)

        keys = samples[0].keys()

        #I should rewrite this when I get more sleep
        batch = OrderedDict()
        for k in keys:
            batch_samples = list()

            for sample in samples:
                sample_d = sample[k]
                sample_d = check_tensor(sample_d)
                sample_d = np.transpose(sample_d,(1,2,3,0))[np.newaxis,:]
                batch_samples.append(sample_d)

            batch[k] = np.concatenate(batch_samples,axis=0)

        return batch


class VolumeDataProvider(DataProvider):
    """
    DataProvider for volumetric data.

    Attributes:
        datasets: List of datasets.
        _sampling_weights: Probability of each dataset being chosen at each
                            iteration.
        _net_spec: Dictionary mapping layers' name to their input dimension.
    """

    def __init__(self, dspec_path, net_spec, params, auto_mask=True):
        """
        Initialize DataProvider.

        Args:
            dspec_path: Path to the dataset specification file.
            net_spec:   Net specification.
            params:     Various options.
            auto_mask:  Whether to automatically generate mask from
                        corresponding label.
        """
        # Params.
        drange = params['drange']            # Required.
        dprior = params.get('dprior', None)  # Optional.

        # Build Datasets.
        print('\n[VolumeDataProvider]')
        p = parser.Parser(dspec_path, net_spec, params, auto_mask=auto_mask)
        self.datasets = list()
        for d in drange:
            print('constructing dataset %d...' % d)
            config, dparams = p.parse_dataset(d)
            dataset = VolumeDataset(config, **dparams)
            self.datasets.append(dataset)

        # Sampling weight.
        self.set_sampling_weights(dprior)

        # Setup data augmentation.
        aug_spec = params.get('augment', [])  # Default is an empty list.
        self._data_aug = DataAugmentor(aug_spec)

    def set_sampling_weights(self, dprior=None):
        """
        TODO(kisuk): Documentation.
        """
        if dprior is None:
            dprior = [x.num_sample() for x in self.datasets]
        # Normalize.
        dprior = np.asarray(dprior, dtype='float32')
        dprior = dprior/np.sum(dprior)
        # Set sampling weights.
        # print 'Sampling weights: {}'.format(['%0.3f' % x for x in dprior])
        self._sampling_weights = dprior

    def next_sample(self):
        """Fetch next sample in a sample sequence."""
        return self.random_sample()  # TODO(kisuk): Temporary.

    def random_sample(self):
        """Fetch random sample."""
        # Pick one dataset randomly.
        dataset = self._get_random_dataset()
        # Draw a random sample and apply data augmenation.
        sample, transform = self._data_aug.random_sample(dataset)
        # Transform sample.
        sample = self._transform(sample, transform)
        # Ensure that sample is ordered by key.
        return OrderedDict(sorted(sample.items(), key=lambda x: x[0]))


    ####################################################################
    ## Private Helper Methods
    ####################################################################

    def _get_random_dataset(self):
        """
        Pick one dataset randomly, according to the given sampling weights.

        Returns:
            Randomly chosen dataset.
        """
        # Trivial case.
        if len(self.datasets)==1:
            return self.datasets[0]

        # Take a single experiment with a multinomial distribution, whose
        # probabilities indicate how likely each dataset be selected.
        # Output is an one-hot vector.
        sq = np.random.multinomial(1, self._sampling_weights, size=1)
        sq = np.squeeze(sq)

        # Get the index of non-zero element.
        idx = np.nonzero(sq)[0][0]

        return self.datasets[idx]

    def _transform(self, sample, transform):
        """
        TODO(kisuk): Documentation.
        """
        affinitized = False
        for key, spec in transform.iteritems():
            if spec is not None:
                if spec['type'] == 'affinitize':
                    affinitized = True
                label_func.evaluate(sample, key, spec)

        # Crop by 1 if affinitized.
        if affinitized:
            for key, data in sample.iteritems():
                sample[key] = tensor_func.crop(data, (1,1,1))

        return sample


class Sampler(object):
    """
    Draw samples from the data provider.

    Attributes:
        dp: Data provider.
        f:  List of sample transformer.
    """

    def __init__(self, dp):
        self.dp = dp
        self.f  = list()

    def __call__(self):
        """Draw a sample, transform if needed."""
        sample = self.dp.random_sample()
        if len(self.f) > 0:
            for f in self.f:
                sample = f(sample)
        return sample

    def set_f(self, f):
        """Reset list, then add a transformer."""
        self.f = list()
        self.add_f(f)

    def add_f(self, f):
        """Add a sample transformer."""
        self.f.append(f)
