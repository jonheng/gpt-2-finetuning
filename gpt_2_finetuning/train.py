import argparse
import json
import os
import numpy as np
import tensorflow as tf
import time
import tqdm
from tensorflow.core.protobuf import rewriter_config_pb2

from .load_dataset import load_dataset, Sampler
from .accumulate import AccumulatingOptimizer
from . import model, sample, encoder, memory_saving_gradients

CHECKPOINT_DIR = 'checkpoint'
SAMPLE_DIR = 'samples'

def maketree(path):
    try:
        os.makedirs(path)
    except:
        pass


def randomize(context, hparams, p):
    if p > 0:
        mask = tf.random.uniform(shape=tf.shape(context)) < p
        noise = tf.random.uniform(shape=tf.shape(context), minval=0, maxval=hparams.n_vocab, dtype=tf.int32)
        return tf.where(mask, noise, context)
    else:
        return context

#TODO: change model_name to model_directory
def train(dataset_path, model_name, n_steps,
          combine=50000, encoding='utf-8',
          batch_size=1, learning_rate=0.00002,
          accumulate_gradients=1, mem_saving_gradients=True,
          only_train_transformer=False, optimizer='adam',
          noise=0.0, top_k=40, top_p=0.0,
          restore_from='latest', run_name='run1',
          sample_every=100, sample_length=1023, 
          sample_num=1, save_every=1000, 
          val_dataset=None, val_batch_size=2,
          val_batch_count=40, val_every=0,
          print_loss_every=1, max_checkpoints_to_keep=5
          ):
    '''
    Training function for GPT-2 models

    Args:
        dataset_path (str):             Input file, directory, or glob pattern (utf-8 text, or preencoded .npz files).
        model_name (str):               Pretrained model name
        n_steps (int):                  Number of training steps
        combine (int):                  Concatenate input files with <|endoftext|> separator into chunks of this minimum size
        encoding (str):                 Set the encoding for reading and writing files.
        batch_size (int):               Batch size
        learning_rate (float):          Learning rate for Adam
        accumulate_gradients (int):     Accumulate gradients across N minibatches.
        mem_saving_gradients (bool):    Use gradient checkpointing to reduce vram usage.
        only_train_transformer (bool):  Restrict training to the transformer blocks.
        optimizer (str):                Optimizer. <adam|sgd>.
        noise (float):                  Add noise to input training data to regularize against typos.
        top_k (int):                    K for top-k sampling.
        top_p (float):                  P for top-p sampling. Overrides top_k if set > 0.
        restore_from (str):             Either "latest", "fresh", or a path to a checkpoint file
        run_name (str):                 Run id. Name of subdirectory in checkpoint/ and samples/
        sample_every (int):             Generate samples every N steps
        sample_length (int):            Sample this many tokens
        sample_num (int):               Generate this many samples
        save_every (int):               Write a checkpoint every N steps
        val_dataset (str):              Dataset for validation loss, defaults to dataset provided in dataset_path.
        val_batch_size (int):           Batch size for validation.
        val_batch_count (int):          Number of batches for validation.
        val_every (int):                Calculate validation loss every given steps.
        print_loss_every (int):         Prints loss every N steps
    '''
    
    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if sample_length > hparams.n_ctx:
        raise ValueError(
            "Can't get samples longer than window size: %s" % hparams.n_ctx)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF
    with tf.Session(config=config) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        context_in = randomize(context, hparams, noise)
        output = model.model(hparams=hparams, X=context_in)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=context[:, 1:], logits=output['logits'][:, :-1]))

        if val_every > 0:
            val_context = tf.placeholder(tf.int32, [val_batch_size, None])
            val_output = model.model(hparams=hparams, X=val_context)
            val_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=val_context[:, 1:], logits=val_output['logits'][:, :-1]))
            val_loss_summary = tf.summary.scalar('val_loss', val_loss)


        tf_sample = sample.sample_sequence(
            hparams=hparams,
            length=sample_length,
            context=context,
            batch_size=batch_size,
            temperature=1.0,
            top_k=top_k,
            top_p=top_p)

        all_vars = [v for v in tf.trainable_variables() if 'model' in v.name]
        train_vars = [v for v in all_vars if '/h' in v.name] if only_train_transformer else all_vars

        if optimizer == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        else:
            exit('Bad optimizer:', optimizer)

        if accumulate_gradients > 1:
            if mem_saving_gradients:
                exit("Memory saving gradients are not implemented for gradient accumulation yet.")
            opt = AccumulatingOptimizer(
                opt=opt,
                var_list=train_vars)
            opt_reset = opt.reset()
            opt_compute = opt.compute_gradients(loss)
            opt_apply = opt.apply_gradients()
            summary_loss = tf.summary.scalar('loss', opt_apply)
        else:
            if mem_saving_gradients:
                opt_grads = memory_saving_gradients.gradients(loss, train_vars)
            else:
                opt_grads = tf.gradients(loss, train_vars)
            opt_grads = list(zip(opt_grads, train_vars))
            opt_apply = opt.apply_gradients(opt_grads)
            summary_loss = tf.summary.scalar('loss', loss)

        summary_lr = tf.summary.scalar('learning_rate', learning_rate)
        summaries = tf.summary.merge([summary_lr, summary_loss])

        summary_log = tf.summary.FileWriter(
            os.path.join(CHECKPOINT_DIR, run_name))

        saver = tf.train.Saver(
            var_list=all_vars,
            max_to_keep=max_checkpoints_to_keep,
            keep_checkpoint_every_n_hours=10000)
        sess.run(tf.global_variables_initializer())

        if restore_from == 'latest':
            ckpt = tf.train.latest_checkpoint(
                os.path.join(CHECKPOINT_DIR, run_name))
            if ckpt is None:
                # Get fresh GPT weights if new run.
                ckpt = tf.train.latest_checkpoint(
                    os.path.join('models', model_name))
        elif restore_from == 'fresh':
            ckpt = tf.train.latest_checkpoint(
                os.path.join('models', model_name))
        else:
            ckpt = tf.train.latest_checkpoint(restore_from)
        print('Loading checkpoint', ckpt)
        saver.restore(sess, ckpt)

        print('Loading dataset...')
        chunks = load_dataset(enc, dataset_path, combine, encoding=encoding)
        data_sampler = Sampler(chunks)
        if val_every > 0:
            if val_dataset:
                val_chunks = load_dataset(enc, val_dataset, combine, encoding=encoding)
            else:
                val_chunks = chunks
        print('dataset has', data_sampler.total_size, 'tokens')
        print('Training...')

        if val_every > 0:
            # Sample from validation set once with fixed seed to make
            # it deterministic during training as well as across runs.
            val_data_sampler = Sampler(val_chunks, seed=1)
            val_batches = [[val_data_sampler.sample(1024) for _ in range(val_batch_size)]
                           for _ in range(val_batch_count)]

        counter = 1
        counter_path = os.path.join(CHECKPOINT_DIR, run_name, 'counter')
        if os.path.exists(counter_path):
            # Load the step number if we're resuming a run
            # Add 1 so we don't immediately try to save again
            with open(counter_path, 'r') as fp:
                counter = int(fp.read()) + 1

        def save(step_count):
            maketree(os.path.join(CHECKPOINT_DIR, run_name))
            print(
                'Saving',
                os.path.join(CHECKPOINT_DIR, run_name,
                             'model-{}').format(step_count))
            saver.save(
                sess,
                os.path.join(CHECKPOINT_DIR, run_name, 'model'),
                global_step=step_count)
            with open(counter_path, 'w') as fp:
                fp.write(str(step_count) + '\n')

        def generate_samples():
            print('Generating samples...')
            context_tokens = data_sampler.sample(1)
            all_text = []
            index = 0
            while index < sample_num:
                out = sess.run(
                    tf_sample,
                    feed_dict={context: batch_size * [context_tokens]})
                for i in range(min(sample_num - index, batch_size)):
                    text = enc.decode(out[i])
                    text = '======== SAMPLE {} ========\n{}\n'.format(
                        index + 1, text)
                    all_text.append(text)
                    index += 1
            print(text)
            maketree(os.path.join(SAMPLE_DIR, run_name))
            with open(
                    os.path.join(SAMPLE_DIR, run_name,
                                 'samples-{}').format(counter), 'w', encoding=encoding) as fp:
                fp.write('\n'.join(all_text))

        def validation():
            print('Calculating validation loss...')
            losses = []
            for batch in tqdm.tqdm(val_batches):
                losses.append(sess.run(val_loss, feed_dict={val_context: batch}))
            v_val_loss = np.mean(losses)
            v_summary = sess.run(val_loss_summary, feed_dict={val_loss: v_val_loss})
            summary_log.add_summary(v_summary, counter)
            summary_log.flush()
            print(
                '[{counter} | {time:2.2f}] validation loss = {loss:2.2f}'
                .format(
                    counter=counter,
                    time=time.time() - start_time,
                    loss=v_val_loss))

        def sample_batch():
            return [data_sampler.sample(1024) for _ in range(batch_size)]


        avg_loss = (0.0, 0.0)
        start_time = time.time()

        try:
            while counter <= n_steps:
                if accumulate_gradients > 1:
                    sess.run(opt_reset)
                    for _ in range(accumulate_gradients):
                        sess.run(
                            opt_compute, feed_dict={context: sample_batch()})
                    (v_loss, v_summary) = sess.run((opt_apply, summaries))
                else:
                    (_, v_loss, v_summary) = sess.run(
                        (opt_apply, loss, summaries),
                        feed_dict={context: sample_batch()})

                summary_log.add_summary(v_summary, counter)

                avg_loss = (avg_loss[0] * 0.99 + v_loss,
                            avg_loss[1] * 0.99 + 1.0)

                if counter % print_loss_every == 0:
                    print(
                        '[{counter} | {time:2.2f}] loss={loss:2.2f} avg={avg:2.2f}'
                        .format(
                            counter=counter,
                            time=time.time() - start_time,
                            loss=v_loss,
                            avg=avg_loss[0] / avg_loss[1]))
                if counter % save_every == 0:
                    save(counter)
                if counter % sample_every == 0:
                    generate_samples()
                if val_every > 0 and (counter % val_every == 0 or counter == 1):
                    validation()

                counter += 1
            if (counter - 1) % save_every != 0 :
                save(counter - 1) 
        except KeyboardInterrupt:
            print('interrupted')
            save(counter)

