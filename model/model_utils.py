from keras.layers import Input, Dropout, Dense, BatchNormalization, Activation, GRU, Embedding, concatenate, Flatten, Lambda
from keras.models import Model


def category_embedding(cat_vars):
    """

    :param cat_vars: cat_vars = [(3,50), (3,50), (3,50), (3,50), (3,50)]--> (num_categories, emb_sz)
    As a rule of thumb (fast.ai): emb_size = min(50, (num_categories+1)/2 ( or (num_diff_levels+1)/2)
    Example: days = 0..6, then emb_sz = min(50, 8/2) = 4
    :type cat_vars:
    :return:
    :rtype:
    """
    # Inputs
    input_l = Input(shape=[len(cat_vars), 1])

    # Category inputs, by slice with Lambda layer.
    # We cannot slice the tensor directly as its output will not be Keras layer.
    category = [Lambda(lambda x: x[:, i])(input_l) for i in range(len(cat_vars))]

    '''
    for i in range(len(cat_vars)):
        category.append(Lambda(lambda x: x[:, i])(input_l))
    '''

    # Apply embedding layers and get emb_outputs
    emb_category = [Embedding(cat_vars[i][0], cat_vars[i][1])(category[i]) for i in range(len(cat_vars))]

    '''
    for i in range(len(cat_vars)):
        emb_category.append(Embedding(cat_vars[i][0], cat_vars[i][1])(category[i]))
    '''



    '''
    concat_l = Flatten()(emb_category[0])
    for i in range(len(cat_vars) - 1):
        concat_l = concatenate([concat_l, Flatten()(emb_category[i + 1])])
    '''

    # We need to flatten since input is len(cat_vars),
    # 1 => so each emb_category.shape = (emb_sz,1), so we need to flatten the extra 1
    emb_outs = [Flatten()(emb_category[i]) for i in range(len(cat_vars))]

    # Concatenated layer
    concat_l = concatenate(emb_outs)
    # TODO: try average pooling, and and learnable (Dense) merge

    # model
    model = Model(input_l, concat_l)

    return model
