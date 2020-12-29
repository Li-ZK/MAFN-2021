import tensorflow as tf
import keras.backend as K
from Utils import ssrn_SS_IN


def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.


# model = ssrn_SS_IN.build_resnet_8()
# .... Define your model here ....
model =ssrn_SS_IN. ResnetBuilder.build_resnet_8
print(get_flops(model))