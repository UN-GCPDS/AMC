import numpy as np
import tensorflow as tf 
import tensorflow_probability as tfp

def _get_random_features_initializer(initializer, shape, seed):
    def _get_cauchy_samples(loc, scale, shape):
        np.random.seed(seed)
        probs = np.random.uniform(low=0., high=1., size=shape)
        return loc + scale * np.tan(np.pi * (probs - 0.5))

    if isinstance(initializer, str):
        if initializer == "gaussian":
            return tf.keras.initializers.RandomNormal(stddev=1.0, seed=seed)
        elif initializer == "laplacian":
            return tf.keras.initializers.Constant(
                _get_cauchy_samples(loc=0.0, scale=1.0, shape=shape))
        else:
            raise ValueError(f'Unsupported kernel initializer {initializer}')
            
class ConvRFF_SinCos(tf.keras.layers.Layer):

    def __init__(self, output_dim, kernel_size=(3,1),
                 scale=None,
                 trainable_scale=False, trainable_W=False,
                 kernel='gaussian',
                 padding='VALID',
                 stride=1,
                 kernel_regularizer=None,
                 normalization=True,
                 seed=None,
                 activation=None,
                 **kwargs):

        super(ConvRFF_SinCos, self).__init__(**kwargs)
        self.output_dim = output_dim  # D/2 in terms of the original formulation
        self.kernel_size = kernel_size
        self.scale = scale
        self.trainable_scale = trainable_scale
        self.trainable_W = trainable_W
        self.padding = padding
        self.stride = stride
        self.initializer = kernel
        self.kernel_regularizer = kernel_regularizer
        self.normalization = normalization
        self.seed = seed
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        kernel_initializer = _get_random_features_initializer(self.initializer,
                                                              shape=(self.kernel_size[0], self.kernel_size[1],
                                                                     input_dim, self.output_dim),
                                                              seed=self.seed)

        self.kernel = self.add_weight(
            name='kernel',
            shape=(self.kernel_size[0], self.kernel_size[1], input_dim, self.output_dim),
            dtype=tf.float32,
            initializer=kernel_initializer,
            trainable=self.trainable_W,
            regularizer=self.kernel_regularizer,
        )

        if not self.scale:
            if self.initializer == 'gaussian':
                self.scale = float(np.sqrt((input_dim * self.kernel_size[0] * self.kernel_size[1]) / 2.0))
            elif self.initializer == 'laplacian':
                self.scale = 1.0
            else:
                raise ValueError(f'Unsupported kernel initializer {self.initializer}')

        self.kernel_scale = self.add_weight( 
            name='kernel_scale',
            shape=(1,),
            dtype=tf.float16,
            initializer=tf.compat.v1.constant_initializer(self.scale),
            trainable=self.trainable_scale,
            constraint=tf.keras.constraints.NonNeg() 
        )
        assert isinstance(self.scale, (float, int)), "self.scale debe ser un número, no una cadena"

    def call(self, inputs):
        scale = tf.math.divide(float(1.0), tf.cast(self.kernel_scale, tf.float32))
        kernel = tf.math.multiply(scale, self.kernel)
        
        # Convolve with kernel for cosine features
        outputs_cos = tf.nn.conv2d(inputs, kernel,
                                   strides=[1, self.stride, self.stride, 1],
                                   padding=self.padding)

        # Convolve with kernel for sine features
        outputs_sin = tf.nn.conv2d(inputs, -kernel,  # negative kernel for sine
                                   strides=[1, self.stride, self.stride, 1],
                                   padding=self.padding)
        
        # Apply sin and cos transformations
        outputs_cos = tf.cos(outputs_cos)
        outputs_sin = tf.sin(outputs_sin)

        # Concatenate the two outputs
        outputs = tf.concat([outputs_sin, outputs_cos], axis=-1)
        output_dim = tf.cast(2 * self.output_dim, tf.float32)  # Factor of 2 for sin and cos

        if self.normalization:
            outputs = tf.math.multiply(tf.math.sqrt(2 / output_dim), outputs)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'kernel_size': self.kernel_size,
            'scale': self.scale,
            'trainable_scale': self.trainable_scale,
            'trainable_W':self.trainable_W,
            'padding':self.padding,
            'kernel':self.initializer,
            'normalization':self.normalization,
            'seed' : self.seed,
            'activation': tf.keras.activations.serialize(self.activation)
        })
        return config
    
class ConvRFF(tf.keras.layers.Layer):

    def __init__(self,output_dim, kernel_size=3,
                 scale=None,
                 trainable_scale=False, trainable_W=False,
                 kernel='gaussian',
                 padding='VALID',
                 stride=1,
                 kernel_regularizer=None,
                 normalization=True,
                 seed=None,
                 mass = False,
                 activation=None,
                 **kwargs):
        
        super(ConvRFF,self).__init__(**kwargs)

        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.scale = scale 
        self.trainable_scale = trainable_scale 
        self.trainable_W = trainable_W
        self.padding = padding
        self.stride = stride
        self.initializer = kernel
        self.kernel_regularizer = kernel_regularizer
        self.normalization = normalization
        self.seed = seed
        self.mass = mass
        self.activation = tf.keras.activations.get(activation)


    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'kernel_size': self.kernel_size,
            'scale': self.scale,
            'trainable_scale': self.trainable_scale,
            'trainable_W':self.trainable_W,
            'padding':self.padding,
            'kernel':self.initializer,
            'normalization':self.normalization,
            'seed' : self.seed,
            'mass': self.mass,
            'activation': tf.keras.activations.serialize(self.activation)
        })
        return config


    def build(self,input_shape):
        input_dim = input_shape[-1]
        kernel_initializer = _get_random_features_initializer(self.initializer,
                                                              shape=(self.kernel_size[0], self.kernel_size[1],
                                                                     input_dim, self.output_dim),
                                                              seed=self.seed)

        self.kernel = self.add_weight(
            name='kernel',
            shape=(self.kernel_size[0], self.kernel_size[1], input_dim, self.output_dim),
            dtype=tf.float32,
            initializer=kernel_initializer,
            trainable=self.trainable_W,
            regularizer=self.kernel_regularizer,
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(self.output_dim,),
            dtype=tf.float32,
            initializer=tf.random_uniform_initializer(
                minval=0.0,maxval=2*np.pi,seed=self.seed),
            trainable=self.trainable_W
        )

        if not self.scale:
            if  self.initializer == 'gaussian':
                self.scale = np.sqrt((input_dim*self.kernel_size**2)/2.0)
            elif self.initializer == 'laplacian':
                self.scale = 1.0
            else: 
                raise ValueError(f'Unsupported kernel initializer {self.initializer}')

        self.kernel_scale = self.add_weight(
            name='kernel_scale',
            shape=(1,),
            dtype=tf.float32,
            initializer=tf.compat.v1.constant_initializer(self.scale),
            trainable=self.trainable_scale,
            constraint='NonNeg'
        )
    

    def _compute_normal_probaility(self,x,mean,std):
        constant = 1/(tf.math.sqrt(2*np.pi)*std)
        return constant*tf.math.exp(-0.5*(x-mean)*(x-mean)/(std*std))


    def _compute_mass(self,):
        weights = tf.reshape(self.kernel,shape=(-1,self.output_dim))
        ww = tf.linalg.norm(weights,axis=0)
        ww_pos = tf.sort(ww)
        mean_pos = tf.reduce_mean(ww_pos)
        std_pos = tf.math.reduce_std(ww_pos)
        mass_pos = self._compute_normal_probaility(ww_pos,mean_pos,std_pos)
        mass_pos = tf.sqrt(tfp.math.trapz(tf.abs(mass_pos),ww_pos))
        return mass_pos


    def call(self,inputs):
        scale = tf.math.divide(1.0, self.kernel_scale)
        kernel = tf.math.multiply(scale,self.kernel)
        outputs = tf.nn.conv2d(inputs,kernel,
                               strides=[1,self.stride,self.stride,1],
                               padding=self.padding)
        outputs = tf.nn.bias_add(outputs,self.bias)
        output_dim = tf.cast(self.output_dim,tf.float32)

        if self.normalization:
            outputs = tf.math.multiply(tf.math.sqrt(2/output_dim),tf.cos(outputs)) 
        else:
            outputs = tf.cos(outputs)
            
        if self.activation is not None:
            outputs = self.activation(outputs)

        outputs = tf.math.multiply(self._compute_mass(),outputs) if self.mass else outputs
        return outputs
    
class ConvRFFC(tf.keras.layers.Layer):

    def __init__(self, output_dim, kernel_size=(3,1),
                 scale=None,
                 trainable_scale=False, trainable_W=False,
                 kernel='gaussian',
                 padding='VALID',
                 stride=1,
                 kernel_regularizer=None,
                 normalization=True,
                 seed=None,
                 mass=False,
                 activation=None,  # New parameter for activation
                 **kwargs):

        super(ConvRFFC, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.scale = scale
        self.trainable_scale = trainable_scale
        self.trainable_W = trainable_W
        self.padding = padding
        self.stride = stride
        self.initializer = kernel
        self.kernel_regularizer = kernel_regularizer
        self.normalization = normalization
        self.seed = seed
        self.mass = mass
        self.activation = tf.keras.activations.get(activation)  # Get the activation function

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'kernel_size': self.kernel_size,
            'scale': self.scale,
            'trainable_scale': self.trainable_scale,
            'trainable_W':self.trainable_W,
            'padding':self.padding,
            'kernel':self.initializer,
            'normalization':self.normalization,
            'seed' : self.seed,
            'mass': self.mass,
            'activation': tf.keras.activations.serialize(self.activation)  # Serialize the activation function
        })
        return config

    def build(self, input_shape):
        input_dim = input_shape[-1]
        kernel_initializer = _get_random_features_initializer(self.initializer,
                                                              shape=(self.kernel_size[0], self.kernel_size[1],
                                                                     input_dim, self.output_dim),
                                                              seed=self.seed)

        self.kernel = self.add_weight(
            name='kernel',
            shape=(self.kernel_size[0], self.kernel_size[1], input_dim, self.output_dim),
            dtype=tf.float32,
            initializer=kernel_initializer,
            trainable=self.trainable_W,
            regularizer=self.kernel_regularizer,
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(self.output_dim,),
            dtype=tf.float32,
            initializer=tf.random_uniform_initializer(
                minval=0.0, maxval=2*np.pi, seed=self.seed),
            trainable=self.trainable_W
        )

        if not self.scale:
            if self.initializer == 'gaussian':
                self.scale = np.sqrt((input_dim * self.kernel_size[0] * self.kernel_size[1]) / 2.0)
            elif self.initializer == 'laplacian':
                self.scale = 1.0
            else:
                raise ValueError(f'Unsupported kernel initializer {self.initializer}')

        self.kernel_scale = self.add_weight(
            name='kernel_scale',
            shape=(1,),
            dtype=tf.float32,
            initializer=tf.compat.v1.constant_initializer(self.scale),
            trainable=self.trainable_scale,
            constraint='NonNeg'
        )

    def _compute_normal_probaility(self, x, mean, std):
        constant = 1 / (tf.math.sqrt(2 * np.pi) * std)
        return constant * tf.math.exp(-0.5 * (x - mean) * (x - mean) / (std * std))

    def _compute_mass(self):
        weights = tf.reshape(self.kernel, shape=(-1, self.output_dim))
        ww = tf.linalg.norm(weights, axis=0)
        ww_pos = tf.sort(ww)
        mean_pos = tf.reduce_mean(ww_pos)
        std_pos = tf.math.reduce_std(ww_pos)
        mass_pos = self._compute_normal_probaility(ww_pos, mean_pos, std_pos)
        mass_pos = tf.sqrt(tfp.math.trapz(tf.abs(mass_pos), ww_pos))
        return mass_pos

    def call(self, inputs):
        scale = tf.math.divide(1.0, self.kernel_scale)
        kernel = tf.math.multiply(scale, self.kernel)
        outputs = tf.nn.conv2d(inputs, kernel,
                               strides=[1, self.stride, self.stride, 1],
                               padding=self.padding)
        outputs = tf.nn.bias_add(outputs, self.bias)
        output_dim = tf.cast(self.output_dim, tf.float32)

        if self.normalization:
            outputs = tf.math.multiply(tf.math.sqrt(2 / output_dim), tf.cos(outputs))
        else:
            outputs = tf.cos(outputs)

        outputs = tf.math.multiply(self._compute_mass(), outputs) if self.mass else outputs

        if self.activation is not None:
            outputs = self.activation(outputs)  # Apply the activation function

        return outputs