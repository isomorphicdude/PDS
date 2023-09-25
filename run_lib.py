import gc
import io
import os
import time

import logging
# Keep the import below for registering the model definitions
import sde_lib
from models import ncsnpp
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
import evaluation


import numpy as np
from absl import flags

import torch
from torchvision.utils import save_image
from utils import restore_checkpoint

import tensorflow as tf
import tensorflow_gan as tfgan



FLAGS = flags.FLAGS


def evaluate(config, workdir, eval_folder, 
             speed_up, freq_mask_path, space_mask_path, 
             alpha, sde_solver_lr=1.2720, 
             verbose=False):
    
    sample_dir = os.path.join(workdir, eval_folder)
    os.makedirs(sample_dir, exist_ok=True)
    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Initialize model
    score_model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # print(checkpoint_dir)

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                            N=int(config.model.num_scales / speed_up))
        sampling_eps = 1e-3 * speed_up
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                               N=int(config.model.num_scales / speed_up))
        sampling_eps = 1e-3 * speed_up
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
                            N=int(config.model.num_scales / speed_up))
        sampling_eps = 1e-5 * speed_up
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    sampling_shape = (config.eval.batch_size,
                      config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps,
                                           freq_mask_path, space_mask_path,alpha, 
                                           sde_solver_lr=sde_solver_lr,
                                           verbose=verbose)
    
    ckpt_path = os.path.join(checkpoint_dir, f'{config.sampling.ckpt_name}')
    state = restore_checkpoint(ckpt_path, state, device=config.device)
    ema.copy_to(score_model.parameters())

    logging.info('start sampling!')
    num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
    
    logging.info('num_sampling_rounds: {}'.format(num_sampling_rounds))
    
    if freq_mask_path is None:
        logging.info(f'freq_mask_path is None, using default mask with lr={sde_solver_lr}')
    if space_mask_path is None:
        logging.info('space_mask_path is None, using default mask')
        
    # samples_ls = []

    for r in range(num_sampling_rounds):
        start = time.time()
        logging.info("sampling -- round: %d" % (r))
        samples, n = sampling_fn(score_model)
        # samples_ls.append(samples)
        for i in range(samples.shape[0]):
            single_sample = samples[i, ...]
            save_image(single_sample.cpu(),
                       os.path.join(sample_dir, 'image_{}.png'.format(i + r * config.eval.batch_size)))

        logging.info('produce one batch of samples')
        logging.info('one batch cost {}'.format(time.time() - start))
        
    # return samples_ls
        
        
def evaluate_fid(config, workdir, eval_folder, 
             speed_up, freq_mask_path, space_mask_path, 
             alpha, sde_solver_lr=1.2720, 
             verbose=False,
             ):
    
    # if data_stats_path is None:
        # raise ValueError('data_stats_path is None')
        
    use_inceptionv3 = config.data.image_size >= 256 # bool
    inception_model = evaluation.get_inception_model(inceptionv3=use_inceptionv3)
    
    sample_dir = os.path.join(workdir, eval_folder)
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Initialize model
    score_model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # print(checkpoint_dir)

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                            N=int(config.model.num_scales / speed_up))
        sampling_eps = 1e-3 * speed_up
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                               N=int(config.model.num_scales / speed_up))
        sampling_eps = 1e-3 * speed_up
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
                            N=int(config.model.num_scales / speed_up))
        sampling_eps = 1e-5 * speed_up
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    sampling_shape = (config.eval.batch_size,
                      config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps,
                                           freq_mask_path, space_mask_path,alpha, 
                                           sde_solver_lr=sde_solver_lr,
                                           verbose=verbose)
    
    ckpt_path = os.path.join(checkpoint_dir, f'{config.sampling.ckpt_name}')
    logging.info(f'ckpt_path: {ckpt_path}')
    state = restore_checkpoint(ckpt_path, state, device=config.device)
    ema.copy_to(score_model.parameters())

    logging.info('start sampling!')
    num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
    
    logging.info('num_sampling_rounds: {}'.format(num_sampling_rounds))
    
    if freq_mask_path is None:
        logging.info(f'freq_mask_path is None, using default mask with lr={sde_solver_lr}')
    if space_mask_path is None:
        logging.info('space_mask_path is None, using default mask')
        
    # detect if there are existing samples to continue
    this_sample_dir = os.path.join(sample_dir, 'samples')
    dir_to_check = tf.io.gfile.glob(os.path.join(this_sample_dir, "statistics_*.npz"))
    if len(dir_to_check) > 0:
        # find the last round
        dir_list = [int(x.split('/')[-1].split('.')[0].split('_')[-1]) for x in dir_to_check]
        last_round = max(dir_list)
        new_round = last_round + 1    
        logging.info(f'Found existing samples in {this_sample_dir}, continue sampling from round {new_round}'
                     )
    else:
        new_round = 0
        logging.info(f'No existing samples found, start sampling from round 0')
    

    for r in range(new_round, num_sampling_rounds):
        start = time.time()
        logging.info("sampling -- round: %d" % (r))
        samples, n = sampling_fn(score_model)
        
        tf.io.gfile.makedirs(this_sample_dir)
        
        samples = np.clip(
                    samples.permute(0, 2, 3, 1).cpu().numpy() * 255.0, 0, 255
                ).astype(np.uint8)

        samples = samples.reshape(
                    (
                        -1,
                        config.data.image_size,
                        config.data.image_size,
                        config.data.num_channels,
                    )
                )
        # Write samples to disk or Google Cloud Storage
        curr_dir = os.path.join(this_sample_dir, f"samples_{r}.npz")
        
        logging.info(f"Writing samples to {curr_dir}")
        
        with tf.io.gfile.GFile(
            curr_dir, "wb"
        ) as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, samples=samples)
            fout.write(io_buffer.getvalue())
            
        # Force garbage collection before calling TensorFlow code for Inception network
        gc.collect()
        
        logging.info("Running Inception network")
        
        latents = evaluation.run_inception_distributed(
            samples, inception_model, inceptionv3=use_inceptionv3
        )
        
        # Force garbage collection again before returning to JAX code
        gc.collect()
        # Save latent represents of the Inception network to disk or Google Cloud Storage
        
        logging.info("Writing statistics")
        
        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, f"statistics_{r}.npz"), "wb"
        ) as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(
                io_buffer, pool_3=latents["pool_3"], logits=latents["logits"]
            )
            fout.write(io_buffer.getvalue())
            
        torch.cuda.empty_cache()

    # Computing FID
    all_logits = []
    all_pools = []    
    
    logging.info("Combining statistics")
    
    stats = tf.io.gfile.glob(os.path.join(this_sample_dir, "statistics_*.npz"))
    for stat_file in stats:
        with tf.io.gfile.GFile(stat_file, "rb") as fin:
            stat = np.load(fin)
            if not use_inceptionv3:
                all_logits.append(stat["logits"])
            all_pools.append(stat["pool_3"])
            
    if not use_inceptionv3:
        all_logits = np.concatenate(all_logits, axis=0)[
                    : config.eval.num_samples
                ]
    all_pools = np.concatenate(all_pools, axis=0)[: config.eval.num_samples]
    
    data_stats = evaluation.load_dataset_stats(config)
    data_pools = data_stats["pool_3"]
    
    logging.info("Computing FID")
    
    fid = tfgan.eval.frechet_classifier_distance_from_activations(
                data_pools, all_pools
            )
    
    logging.info("FID: %.6e" % fid)
    
    
