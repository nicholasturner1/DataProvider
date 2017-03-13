#!/usr/bin/env python
__doc__ = """

Dataset classes.

Kisuk Lee <kisuklee@mit.edu>, 2016
"""

from collections import OrderedDict
import copy
import numpy as np

from box import Box
from config_data import ConfigData, ConfigLabel
from vector import Vec3d

#new
from tensor import TensorData
from transform import *
from utils import *

class Dataset(object):
    """
    Dataset interface.
    """

    def next_sample(self):
        raise NotImplementedError

    def random_sample(self):
        raise NotImplementedError


class SampleDataset(Dataset):
    """
    Dataset for a full sample. Assumes that the user (or SampleDataProvider)
    has done most of the work to massage the data into a proper form for sampling,
    but will expand scalar datasets (usually labels) to fill the net spec.

    Attributes:
        _data:  Dictionary mapping layer's name to TensorData, which contains
                4D volumetric data (e.g. EM image stacks, label stacks, etc.).

        _image: List of image layers' names.

        _label: List of label layers' names.

        _spec:  Net specification. Dictionary mapping layer's name to its input
                dimension (either 3-tuple or 4-tuple).

        _range: Range of valid coordinates for accessing data given a net spec.
                It depends both on data and net specs.

        params: Dataset-specific parameters.
    """

    def __init__(self, data, layer_types, net_spec, preps, xforms, **kwargs):
        """__init__(self, data, layer_types, net_spec, preps)"""

        #Copying references
        data = { k : v for (k,v) in data.items() }

        # Set dataset-specific params.
        self.params = dict()
        for k, v in kwargs.items():
            self.params[k] = v

        # fills self._image and self._label
        self._assign_dsets(layer_types)

        # performs preprocessing on the data volumes in self._image
        self._preprocess_images(data, self._image, preps)

        # ensures each dataset can fulfill the spec, and fills the _spec attr
        self._fulfill_net_spec(data, net_spec)
        
        # fills the _data attribute
        self._fill_data(data, net_spec)
        self._update_range()

        self._fill_xforms(xforms, self._label)


    def next_sample(self, spec=None):
        """Fetch next sample in a sample sequence."""
        return self.random_sample(spec)  # Currently just pick randomly.


    def random_sample(self, spec=None):
        """Fetch sample randomly"""
        # Dynamically change spec.
        if spec is not None and spec != self._spec:
          print("No increased sample implemented for SampleDataset!")
          raise NotImplementedError

        pos = self._random_location()
        ret = self.get_sample(pos)

        # ret is a 2-tuple (sample, transform).
        return ret

    def get_sample(self, pos):
        """Extract a sample centered on pos.

        Every data in the sample is guaranteed to be center-aligned.

        Args:
            pos: Center coordinate of the sample.

        Returns:
            sample:     Dictionary mapping input layer's name to data.
            transform:  Dictionary mapping label layer's name to the type of
                        label transformation specified by user.

        """
        # self._spec is guaranteed to be ordered by key, so using OrderedDict
        # and iterating over already-sorted self._spec together guarantee the
        # sample is sorted.
        sample = OrderedDict()
        for name in self._spec.keys():
            sample[name] = self._data[name].get_patch(pos)

        transform = dict()
        for name in self._label:
            transform[name] = self._xforms[name]

        return sample, transform

    ####################################################################
    ## Private Helper Methods
    ####################################################################

    def _assign_dsets(self, layer_types):
        """ Assigns image and label volumes to their category """

        self._image, self._label = list(), list()

        for (name,t) in layer_types.items():
            if "_mask" in t:
              continue
            elif "label" in t:
              self._label.append(name)
            else:
              self._image.append(name)


    def _preprocess_images(self, data, images, preps):
        """ Performs preprocessing on the listed data volumes """
        
        for pp in preps:
            assert(isinstance(pp,dict))
            assert("type" in pp.keys())

        for im in images:
            data[im] = check_tensor(data[im])

            for pp in preps:
                data[im] = tensor_func.evaluate(data[im],pp)


    def _fulfill_net_spec(self, data, net_spec):
        """ Ensures that each dataset can fulfill the net spec, and
        fills the _spec attr """
        
        #error checking
        for name in net_spec.keys():
            assert(name in data.keys())


        for name,sz in net_spec.items():
            d = data[name]
            if isinstance(d,np.ndarray):
              if d.size == 1:
                d = self._expand_data(d,sz)
              else:
                assert(d.shape[-3:] == sz)
            else: #
              assert(isinstance(d,(float,int,np.float32,np.uint32)))
              d = self._expand_data(d,sz)

            data[name] = d

        self._spec = net_spec


    def _expand_data(self, d, sz):

        if isinstance(d,(float,np.float32,np.float64)):
            res = np.empty(sz,np.float)
            res[:] = d

        elif isinstance(d,(int,np.uint32)):
            res = np.empty(sz,np.int)
            res[:] = d

        else:
            raise NotImplementedError

        return res


    def _fill_data(self, data, net_spec):
        """ Fills the _data attr """
        max_shape = (0,0,0)
        for k,v in net_spec.items():
            if np.prod(v) > np.prod(max_shape):
              max_shape = v

        max_shape = np.array(max_shape)
        #volumes are assumed to be aligned to a center coord
        offsets = { k : tuple( (max_shape - v)//2 ) for k,v in net_spec.items() }

        self._data = { name : TensorData(d,net_spec[name],offsets[name]) for
                       name,d in data.items() }


    def _fill_xforms(self, xforms, labels):
        
        for l in labels:
          assert(l in xforms.keys())

        self._xforms = xforms


    def _update_range(self):
        """
        Update valid range. It's computed by intersecting the valid range of
        each TensorData.
        """
        self._range = None
        for name, dim in self._spec.items():
            # Update patch size.
            self._data[name].set_fov(dim[-3:])
            # Update valid range.
            r = self._data[name].range()
            if self._range is None:
                self._range = r
            else:
                self._range = self._range.intersect(r)


    def _random_location(self):
        """Return one of the valid locations randomly."""
        s = self._range.size()
        z = np.random.randint(0, s[0])
        y = np.random.randint(0, s[1])
        x = np.random.randint(0, s[2])
        # Global coordinate system.
        return Vec3d(z,y,x) + self._range.min()
        # DEBUG
        #return self._range.min()


class VolumeDataset(Dataset):
    """
    Dataset for volumetric data.

    Attributes:
        _data:  Dictionary mapping layer's name to TensorData, which contains
                4D volumetric data (e.g. EM image stacks, label stacks, etc.).

        _image: List of image layers' names.

        _label: List of label layers' names.

        _spec:  Net specification. Dictionary mapping layer's name to its input
                dimension (either 3-tuple or 4-tuple).

        _range: Range of valid coordinates for accessing data given a net spec.
                It depends both on data and net specs.

        params: Dataset-specific parameters.
    """

    def __init__(self, config, **kwargs):
        """Initialize VolumeDataset."""
        self.build_from_config(config)
        # Set dataset-specific params.
        self.params = dict()
        for k, v in kwargs.items():
            self.params[k] = v

    def build_from_config(self, config):
        """
        Build dataset from a ConfiParser object generated by Parser's
        parse_dataset method.
        """
        self._reset()

        # First pass for images and labels.
        assert config.has_section('dataset')
        for name, data in config.items('dataset'):
            assert config.has_section(data)
            if '_mask' in data:
                # Mask will be processed later.
                continue
            if 'label' in data:
                self._data[name] = ConfigLabel(config, data)
                self._label.append(name)
            else:
                self._data[name] = ConfigData(config, data)
                self._image.append(name)

        # Second pass for masks.
        for name, data in config.items('dataset'):
            if '_mask' in data:
                if config.has_option(data, 'shape'):
                    # Lazy filling of mask shape. Since the shape of mask should
                    # be the same as the shape of corresponding label, it can be
                    # known only after having processed label in the first pass.
                    label = data.replace('_mask',"")
                    shape = self._data[label].shape()
                    config.set(data, 'shape', shape)
                self._data[name] = ConfigData(config, data)

        # Set dataset spec.
        spec = dict()
        for name, data in self._data.items():
            spec[name] = tuple(data.fov())
        self.set_spec(spec)

    def get_spec(self):
        """Return dataset spec."""
        # TODO(kisuk):
        #   spec's value type is tuple, which is immutable. Do we still need to
        #   deepcopy it?
        return copy.deepcopy(self._spec)

    def get_imgs(self):
        return copy.deepcopy(self._image)

    def set_spec(self, spec):
        """Set spec and update valid range."""
        # Order by key
        self._spec = OrderedDict(sorted(spec.items(), key=lambda x: x[0]))
        self._update_range()

    def num_sample(self):
        """Return number of samples in valid range."""
        s = self._range.size()
        return s[0]*s[1]*s[2]

    def get_range(self):
        """Return valid range."""
        return Box(self._range)

    def get_sample(self, pos):
        """Extract a sample centered on pos.

        Every data in the sample is guaranteed to be center-aligned.

        Args:
            pos: Center coordinate of the sample.

        Returns:
            sample:     Dictionary mapping input layer's name to data.
            transform:  Dictionary mapping label layer's name to the type of
                        label transformation specified by user.

        """
        # self._spec is guaranteed to be ordered by key, so using OrderedDict
        # and iterating over already-sorted self._spec together guarantee the
        # sample is sorted.
        sample = OrderedDict()
        for name in self._spec.keys():
            sample[name] = self._data[name].get_patch(pos)

        transform = dict()
        for name in self._label:
            transform[name] = self._data[name].get_transform()

        return sample, transform

    def next_sample(self, spec=None):
        """Fetch next sample in a sample sequence."""
        return self.random_sample(spec)  # Currently just pick randomly.

    def random_sample(self, spec=None):
        """Fetch sample randomly"""
        # Dynamically change spec.
        if spec is not None:
            original_spec = self._spec
            try:
                self.set_spec(spec)
            except:
                self.set_spec(original_spec)
                raise

        pos = self._random_location()
        ret = self.get_sample(pos)

        # Return to original spec.
        if spec is not None:
            self.set_spec(original_spec)

        # ret is a 2-tuple (sample, transform).
        return ret

    ####################################################################
    ## Private Helper Methods
    ####################################################################

    def _reset(self):
        """Reset all attributes."""
        self._data  = dict()
        self._image = list()
        self._label = list()
        self._spec  = None
        self._range = None

    def _random_location(self):
        """Return one of the valid locations randomly."""
        s = self._range.size()
        z = np.random.randint(0, s[0])
        y = np.random.randint(0, s[1])
        x = np.random.randint(0, s[2])
        # Global coordinate system.
        return Vec3d(z,y,x) + self._range.min()
        # DEBUG
        #return self._range.min()

    def _update_range(self):
        """
        Update valid range. It's computed by intersecting the valid range of
        each TensorData.
        """
        self._range = None
        for name, dim in self._spec.items():
            # Update patch size.
            self._data[name].set_fov(dim[-3:])
            # Update valid range.
            r = self._data[name].range()
            if self._range is None:
                self._range = r
            else:
                self._range = self._range.intersect(r)
