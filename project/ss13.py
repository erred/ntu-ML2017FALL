import csv
import os
import pickle
import sys

import tensorflow as tf

import iofn

# =================== Hyperparams
BATCHSIZE = 256
EPOCHS = 200
EMBED = 16
LEARNING_RATE = 0.001
SHARED_UNITS = 512
MEMORY_SIZE = 512
LOGSTEPS = 160
EVALSTEPS = 15
# =================== Constants
# RECORDS = 45036

VOCAB = 2389 + 4 + 1
MAX_OUTPUT_LEN = 16

PAD_TOKEN = 0
START_TOKEN = 1
END_TOKEN = 2
UNKONW_TOKEN = 3

VERSION = os.path.basename(os.path.splitext(sys.argv[0])[0])
MODELDIR = 'model/' + VERSION
tf.logging.set_verbosity(tf.logging.INFO)


# =================== Define Model
def model_fn(features, labels, mode):

    # =================== Inputs
    inputs = tf.cast(features['x'], tf.float32)
    seqlens = tf.cast(features['seqlens'], tf.int64)
    lens = tf.cast(features['lens'], tf.int32)
    batch_size = tf.shape(inputs)[0]

    embeddings = tf.get_variable(
        "embeddings", shape=(VOCAB, EMBED), dtype=tf.float32)

    training = False
    if mode == tf.estimator.ModeKeys.TRAIN:
        training = True

    if mode != tf.estimator.ModeKeys.PREDICT:
        embedLabels = tf.nn.embedding_lookup(embeddings, labels)
    else:
        choices = tf.cast(features['choices'], tf.int64)

    # =================== Encoder
    encoder_cell1 = tf.contrib.rnn.GRUCell(num_units=SHARED_UNITS // 2)
    encoder_cell2 = tf.contrib.rnn.GRUCell(num_units=SHARED_UNITS // 2)
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=encoder_cell1,
        cell_bw=encoder_cell2,
        inputs=inputs,
        sequence_length=seqlens,
        swap_memory=True,
        dtype=tf.float32)

    encoder_output = tf.concat(encoder_output, 2)
    encoder_state = tf.concat(encoder_state, -1)

    # ==================== Decoder
    decoder_cell = tf.contrib.rnn.GRUCell(num_units=SHARED_UNITS)

    # ==================== Attention
    attn_mech = tf.contrib.seq2seq.BahdanauAttention(
        num_units=SHARED_UNITS, memory=encoder_output, normalize=True)
    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
        cell=decoder_cell,
        attention_mechanism=attn_mech,
        attention_layer_size=MEMORY_SIZE)
    decoder_initial_state = decoder_cell.zero_state(
        dtype=tf.float32, batch_size=batch_size)
    decoder_initial_state = decoder_initial_state.clone(
        cell_state=encoder_state)

    # ==================== Helper
    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        embedding=embeddings,
        start_tokens=tf.tile([START_TOKEN], [batch_size]),
        end_token=END_TOKEN)
    if mode == tf.estimator.ModeKeys.TRAIN:
        prob = tf.train.inverse_time_decay(1.0, tf.train.get_global_step(),
                                           4000, 0.5)
        helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
            inputs=embedLabels,
            sequence_length=lens,
            embedding=embeddings,
            sampling_probability=prob)

    # ==================== Decoder
    decoder = tf.contrib.seq2seq.BasicDecoder(
        cell=decoder_cell, helper=helper, initial_state=decoder_initial_state)

    decode_output, _, output_len = tf.contrib.seq2seq.dynamic_decode(
        decoder=decoder,
        impute_finished=True,
        maximum_iterations=MAX_OUTPUT_LEN)
    output = decode_output.rnn_output

    # ==================== Pad output
    outdims = output.get_shape()[2]

    def padOutput(elems):
        tshape = (MAX_OUTPUT_LEN, outdims)
        paddings = [[0, MAX_OUTPUT_LEN - tf.shape(elems)[0]], [0, 0]]
        return tf.reshape(tf.pad(elems, paddings), tshape)

    output = tf.map_fn(fn=padOutput, elems=output)

    # =================== Output projection
    output = tf.reshape(output, [-1, outdims])
    output = tf.layers.dropout(output, training=training)
    output = tf.layers.dense(output, VOCAB, activation=tf.nn.relu)
    output = tf.reshape(output, [-1, MAX_OUTPUT_LEN, VOCAB])

    # =================== Model Ends Here
    if mode == tf.estimator.ModeKeys.PREDICT:
        return modelSpec(mode, output, output_len=output_len, choices=choices)
        # return modelSpec(mode, output, output_len=output_len)
    return modelSpec(mode, output, labels=labels, output_len=lens)


# =================== Make TF run
def modelSpec(mode,
              output,
              ids=None,
              labels=None,
              output_len=None,
              choices=None):
    if mode == tf.estimator.ModeKeys.PREDICT:
        print(output, choices)
        mask = tf.cast(
            tf.sequence_mask(output_len, MAX_OUTPUT_LEN), tf.float32)
        loss = tf.contrib.seq2seq.sequence_loss(
            logits=tf.contrib.seq2seq.tile_batch(output, 4),
            targets=tf.reshape(choices, [-1, 16]),
            weights=tf.contrib.seq2seq.tile_batch(mask, 4),
            average_across_batch=False)
        loss = tf.reshape(loss, [-1, 4])
        loss = tf.argmin(loss, 1)
        print(loss)

        predictions = {
            'seq': tf.argmax(output, 2),
            'lens': output_len,
            'choice': loss
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    else:
        mask = tf.cast(
            tf.sequence_mask(output_len, MAX_OUTPUT_LEN), tf.float32)
        loss = tf.contrib.seq2seq.sequence_loss(
            logits=output, targets=labels, weights=mask)
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            learning_rate=LEARNING_RATE,
            optimizer=optimizer)
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)


if __name__ == "__main__":

    runConfig = tf.estimator.RunConfig()
    runConfig = runConfig.replace(
        log_step_count_steps=50,
        keep_checkpoint_max=2,
        save_checkpoints_steps=LOGSTEPS,
        save_summary_steps=LOGSTEPS)
    estimator = tf.estimator.Estimator(
        model_fn=model_fn, params=None, config=runConfig, model_dir=MODELDIR)

    train_fn, eval_fn = iofn.trainFn(batch_size=BATCHSIZE, epochs=EPOCHS)
    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_fn,
        eval_input_fn=eval_fn,
        eval_steps=EVALSTEPS,
        eval_delay_secs=1,
        min_eval_frequency=1)

    experiment.train_and_evaluate()

    test_fn, texts = iofn.testFn(batch_size=BATCHSIZE)
    preds = estimator.predict(input_fn=test_fn)
    preds = [[p['seq'][:p['lens']], p['choice']] for p in preds]
    # print([p[1] for p in preds])
    with open('out/' + VERSION, 'w') as f:
        w = csv.writer(f)
        w.writerow(['id', 'answer'])
        w.writerows([[i + 1, p[1]] for i, p in enumerate(preds)])

    with open('i2w.pkl', 'rb') as f:
        i2w = pickle.load(f)

    for pred in preds:
        out = [i2w[p] for p in pred[0].tolist()]
        print(' '.join(out))
