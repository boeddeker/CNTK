# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
"""
Tensor operations.
"""


import warnings
import functools
import numpy as np
from scipy import sparse

import cntk


def _consecutive(data):
    """
    https://stackoverflow.com/a/7353335/5766934

    >>> a = np.array([0, 47, 48, 49, 50, 97, 98, 99])
    >>> _consecutive(a)
    [[0], [47, 48, 49, 50], [97, 98, 99]]
    >>> a = [4, 5, 6, 7, 8]
    >>> _consecutive(a)
    [[4, 5, 6, 7, 8]]
    >>> type(_consecutive(a)[0][0])
    <class 'int'>
    """
    assert len(data) > 0
    return list(map(lambda x: list(map(int, x)), np.split(data, np.where(np.diff(data) != 1)[0] + 1)))


class TensorOpsMixin(object):
    '''
    This class defines math overloads so that CNTK nodes can be written in math
    expressions.
    '''

    # operator overload for (+) where self is the left operand
    def __add__(self, other):
        if other is 0 or other is 0.:
            return self
        from . import ops
        return ops.plus(self, other)

    # operator overload for (+) where self is the right operand
    def __radd__(self, other):
        if other is 0 or other is 0.:
            return self
        from . import ops
        return ops.plus(other, self)

    # operator overload for (-) where self is the left operand
    def __sub__(self, other):
        if other is 0 or other is 0.:
            return self
        from . import ops
        return ops.minus(self, other)

    # operator overload for (-) where self is the right operand
    def __rsub__(self, other):
        if other is 0 or other is 0.:
            return -self
        from . import ops
        return ops.minus(other, self)

    # operator overload for (*) where self is the left operand
    def __mul__(self, other):
        if other is 0 or other is 0.:
            return 0
        if other is 1 or other is 1.:
            return self
        from . import ops
        return ops.element_times(self, other)

    # operator overload for (*) where self is the right operand
    def __rmul__(self, other):
        if other is 0 or other is 0.:
            return 0
        if other is 1 or other is 1.:
            return self
        from . import ops
        return ops.element_times(other, self)

    # operator overload for (@) where self is the left operand
    def __matmul__(self, other):
        # NOTE supported in Python 3.5
        from . import ops
        return ops.times(self, other)

    # operator overload for (@) where self is the right operand
    def __rmatmul__(self, other):
        # NOTE supported in Python 3.5
        from . import ops
        return ops.times(other, self)

    # operator overload for (/) where self is the left operand
    def __truediv__(self, other):
        from . import ops
        self.__div__ = self.__truediv__
        return ops.element_divide(self, other)

    # operator overload for (/) where self is the right operand
    def __rtruediv__(self, other):
        from . import ops
        self.__rdiv__ = self.__rtruediv__
        return ops.element_divide(other, self)

    # Python2 compatibility
    __div__ = __truediv__
    __rdiv__ = __rtruediv__

    def __abs__(self):
        from . import ops
        return ops.abs(self)

    def __neg__(self):
        from . import ops
        return ops.negate(self)

    # TODO __xor__, __rxor__, __pow__, __rpow__,  __invert__

    # Comparison operators are not exposed yet, because of __eq__ being
    # required to allow comparison of Variables on C++ so that we can say
    # 'for var in variables'.
    # __lt__, __le__, __gt__, __ge__, __and__, __rand__, __or__, __ror__,

    def __getitem__(self, arg):
        """
    
        Costs:
        None -> one reshape
        int -> one reshape
        slice -> shared slice
        tuple of consecutive idx -> shared slice
        tuple of non consecutive idx -> one slice per consecutive part and a splice
    
        """

        if hasattr(self, 'outputs') and len(self.outputs) > 1:
            try:
                return self.outputs[arg]
            except Exception as e:
                msg = 'Slice for multioutput functions is not supported, ' \
                      'the fallback to select to output requires ' \
                      'that only one index is provided. arg: {}, self: {}'.format(
                    arg, self)
                raise KeyError(msg) from e

        # ToDo: shape check for int and tuple
        if not isinstance(arg, tuple):
            arg = (arg,)

        # print(arg)

        axis = []
        begin_index = []
        end_index = []
        cur_axis = -1

        count_ellipsis = sum([a is Ellipsis for a in arg])

        count_tuple = sum([isinstance(a, (tuple, list)) for a in arg])

        if not count_ellipsis <= 1:
            raise IndexError("an index can only have a single ellipsis ('...')")

        if not count_tuple <= 1:
            raise IndexError("Advance slicing is only partially supported, can only have a single tuple or list")

        nones = sum([a is None for a in arg])

        if len(arg) - nones - count_ellipsis > len(self.shape):
            raise IndexError(len(arg), nones, count_ellipsis, len(self.shape))

        reshapes = []
        insert_axis = []

        ndim = len(self.shape) + nones

        for a in arg:
            cur_axis += 1
            #         print('a', a)
            if a is Ellipsis:
                cur_axis -= len(arg)
            elif isinstance(a, slice):
                #             print(a.start, a.step, a.stop)
                if (a.start, a.step, a.stop) == (None, None, None):
                    pass
                else:
                    axis.append(cur_axis)
                    begin_index.append(a.start or 0)
                    end_index.append(a.stop or 0)
            elif isinstance(a, int):
                axis.append(cur_axis)
                begin_index.append(a)
                end_index.append(a + 1)

                reshapes.append(functools.partial(cntk.reshape, shape=tuple(), begin_axis=cur_axis % ndim,
                                                  end_axis=cur_axis % ndim + 1))

            elif a is None:
                #             print(cur_axis)
                # self = cntk.reshape(self, shape=(1,), begin_axis=cur_axis % ndim, end_axis=cur_axis % ndim)

                insert_axis += [cur_axis % ndim]

            elif isinstance(a, (tuple, list)):
                # Select multiple elements from the same dimension. This is
                # different from NumPy's advanced indexing, since we just go
                # axis by axis from left to right and don't do any
                # broadcasting.

                if not all(isinstance(idx, int) for idx in a):
                    raise IndexError(
                        'indices have to be of type int.'
                        + str([isinstance(idx, int) for idx in a]))

                slice_accum = []

                consecutive_idx_list = _consecutive(a)
                if len(consecutive_idx_list) <= 1:
                    axis.append(cur_axis)
                    begin_index.append(consecutive_idx_list[0][0])
                    end_index.append(consecutive_idx_list[0][-1] + 1)
                else:
                    for cons_indices in consecutive_idx_list:
                        slice_accum.append(cntk.ops.slice(self, axis=cur_axis % ndim,
                                                          begin_index=cons_indices[0],
                                                          end_index=cons_indices[-1] + 1))
                    self = cntk.ops.splice(*slice_accum, axis=cur_axis)
            else:
                raise ValueError(a)

                #     print('axis', axis, type(axis), [type(i) for i in axis])
                #     print('begin_index', begin_index, type(begin_index), [type(i) for i in begin_index])
                #     print('end_index', end_index, type(end_index), [type(i) for i in end_index])

        if len(insert_axis):
            consecutive_idx_list = _consecutive(insert_axis)
            for i in consecutive_idx_list:
                self = cntk.reshape(self, shape=tuple((1,)*len(i)), begin_axis=i[0],
                                    end_axis=i[0])

        if len(axis) > 0:
            self = cntk.ops.slice(self, axis=axis, begin_index=begin_index, end_index=end_index)

        for r in reversed(reshapes):
            self = r(self)

        # assert len(reshapes) == 1, "Currently is only one np.newaxis/None allowed"

        return self


AVAILABLE_TENSOR_OPS = ['abs', 'add', 'div', 'getitem', 'matmul', 'mul',
                        'radd', 'rdiv', 'rmatmul', 'rmul', 'rsub', 'rtruediv',
                        'sub', 'truediv', 'neg']


def _add_tensor_ops(klass):
    for op_name in AVAILABLE_TENSOR_OPS:
        overload_name = '__%s__' % op_name

        if getattr(klass, overload_name, None):
            raise ValueError('class "%s" already has operator overload "%s"' %
                             (klass, overload_name))

        setattr(klass, overload_name, TensorOpsMixin.__dict__[overload_name])


class ArrayMixin(object):
    def asarray(self):
        '''
        Converts the instance's data to a NumPy array.
        '''
        import cntk
        result = None
        if isinstance(self, cntk.Constant):
            ndav = super(cntk.Constant, self).value()
            is_sparse = ndav.is_sparse()
        elif isinstance(self, cntk.Parameter):
            ndav = super(cntk.Parameter, self).value()
            is_sparse = ndav.is_sparse()
        elif isinstance(self, (cntk.cntk_py.Constant, cntk.cntk_py.Parameter)):
            ndav = self.value()
            is_sparse = ndav.is_sparse()

        elif isinstance(self, (cntk.cntk_py.NDArrayView, cntk.cntk_py.NDMask)):
            ndav = self
            if isinstance(self, cntk.NDArrayView):
                is_sparse = ndav.is_sparse
            elif isinstance(self, cntk.cntk_py.NDArrayView):
                is_sparse = ndav.is_sparse()
            else:
                is_sparse = False

        # Value and MinibatchData have a mask, which means that we need the
        # corresponding Variable to do the proper conversion. For easy
        # discoverability, we nevertheless add asarray() to those classes as
        # well, but issue a warning.
        elif isinstance(self, cntk.cntk_py.Value) or isinstance(self, cntk.cntk_py.MinibatchData):

            if isinstance(self, cntk.cntk_py.MinibatchData):
                value = self.data
            else:
                value = self

            if isinstance(value, cntk.Value):
                is_sparse = value.is_sparse
                has_mask = super(cntk.Value, value).mask() is not None
                ndav = value.data
            else:
                is_sparse = value.is_sparse()
                has_mask = value.mask() is not None
                ndav = value.data()

            if has_mask:
                warnings.warn('asarray() will ignore the mask information. '
                              'Please use as_sequences() to do the proper '
                              'conversion.')

        if is_sparse:
            from cntk.internal.sanitize import _sparse_to_dense_network_cache

            device = ndav.device
            if callable(device):
                device = device()

            network = _sparse_to_dense_network_cache(ndav.shape[1:], False,
                                                     device)
            warnings.warn('converting Value object to CSR format might be slow')

            dense_data = network.eval(self, device=device)

            def to_csr(dense_data):
                if len(dense_data.shape) > 2:
                    raise ValueError('Cannot convert a sparse NDArrayView or Value object '
                                     'with shape %s of rank > 2 to a scipy.csr matrix.' % str(dense_data.shape))
                return sparse.csr_matrix(dense_data)

            if isinstance(dense_data, list):
                result = [to_csr(d) for d in dense_data]
            else:
                result = to_csr(dense_data)

        else:
            result = ndav.to_ndarray()

        return result

def _add_asarray(klass):
    member_name = 'asarray'

    if getattr(klass, member_name, None):
        raise ValueError('class "%s" has already an attribute "%s"' %
                         (klass, member_name))

    setattr(klass, member_name, ArrayMixin.__dict__[member_name])
