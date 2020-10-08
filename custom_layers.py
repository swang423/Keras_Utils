from keras import backend as K
from keras.engine.topology import Layer
from keras.layers.merge import _Merge
import numpy as np
import tensorflow as tf
from keras.layers import Lambda
from keras.initializers import RandomNormal,VarianceScaling, Constant
from keras.constraints import NonNeg
from keras import initializers


class ArgMax(Layer):
    """Simulate ArgMax-OneHot so that only the max dimension is 1, rest is 0."""
    def __init__(self,**kwargs):
        super(ArgMax,self).__init__(**kwargs)
    def call(self,inputs):
        return tf.sign(X-tf.reduce_max(inputs,axis=-1,keepdims=True))+1

class Frame2Lps(Layer):
    def __init__(self,fftSize=512,**kwargs):
        self.fftSize = fftSize
        self.fft_length = K.constant(np.asarray([fftSize]),dtype='int32')
        self.window = K.constant(np.hamming(fftSize))
        super(Frame2Lps,self).__init__(**kwargs)
    def build(self,input_shape):
        assert input_shape[1] == self.fftSize
        super(Frame2Lps,self).build(input_shape)
    def call(self,x):
        windowedFeat = x*self.window
        fft = tf.spectral.rfft(windowedFeat,fft_length=self.fft_length)
        mag = tf.abs(fft)
        magClip = K.clip(mag,1e-15,None)
        return K.log(magClip)*2 #*2 because we used mag not energy
    def compute_output_shape(self, input_shape):
        return input_shape[0],int(self.fftSize/2+1)


class Lps2Frame(_Merge):
    # Use GlobalCmvn first to convert denorm lps feats
    def __init__(self,windowSize=0,**kwargs):
        self._reshape_required = True
        if windowSize > 0:
            self.window = K.constant(np.hamming(windowSize))
        else:
            self.window = None
        super(Lps2Frame,self).__init__(**kwargs)
    def build(self, input_shape):
        #Used purely for shape validation.
        if not isinstance(input_shape, list):
            raise ValueError('A `Lps2Frame` layer should be called '
                             'on two inputs.')
        assert(len(input_shape)==2),"Require two inputs"
        assert(input_shape[0] == input_shape[1])
    def _merge_function(self, inputs):
        #1st item in list is magnitude
        #2nd item in list is phase
        lps = inputs[0]
        phs = inputs[1]
        mag = K.exp(lps/2)
        real = mag * tf.cos(phs)
        imag = mag * tf.sin(phs)
        stft = tf.complex(real,imag)
        istft = Lambda(lambda v:tf.to_float(tf.spectral.irfft(tf.cast(v,dtype=tf.complex64))))(stft)
        if self.window is not None:
            istft = istft/self.window
        return istft
    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('A `Lps2Frame` layer should be called '
                             'on two inputs.')
        dim_fft_half = input_shape[0][1]-1
        return input_shape[0][0],dim_fft_half*2
    class Irfft(Layer):
    def __init__(self,**kwargs):
        super(Irfft,self).__init__(**kwargs)
    def build(self,input_shape):
        super(Irfft,self).build(input_shape)
    def call(self,x):
        x_amp = K.exp(x[...,0])
        x_phs = x[...,1]
        x_cmplx = tf.complex(x_amp,x_phs)
        return tf.spectral.irfft(x_cmplx)
    def compute_output_shape(self, input_shape):
        assert len(input_shape)== 3
        return (input_shape[0],(input_shape[1]-1) * 2)
  class Framing(Layer):
    def __init__(self,slice=11,**kwargs):
        self.slice = slice
        super(Framing, self).__init__(**kwargs)
    def build(self, input_shape):
        super(Framing, self).build(input_shape)
    def call(self,x):
        nrow,ncol = K.int_shape(x)
#        out = K.placeholder(nrow-self.slice+1,self.slice,ncol)
        out = []
        beg = 0
        for kk in range(nrow-self.slice+1):
            out.append(K.expand_dims(x[beg:beg+self.slice-1,:],axis=0))
            beg += 1
        out = K.concatenate(out,axis=0)
        return out
    def compute_output_shape(self, input_shape):
        assert len(input_shape)==2
        return (input_shape[0]-self.slice+1,self.slice,input_shape[1])
