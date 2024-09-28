import os
import argparse
import json
from typing import Optional, Union
import numpy as np
import torch
from lightning_fabric import seed_everything


from utils.util import get_mask_mnr, get_mask_bm, get_mask_rm
from utils.util import find_max_epoch, print_size, sampling, calc_diffusion_hyperparams
from utils.data_factory import data_provider

from imputers.DiffWaveImputer import DiffWaveImputer
from imputers.SSSDSAImputer import SSSDSAImputer
from imputers.SSSDS4Imputer import SSSDS4Imputer

from sklearn.metrics import mean_squared_error
from statistics import mean
import matplotlib.pyplot as plt


def mse(
    y: np.ndarray,
    y_hat: np.ndarray,
    weights: Optional[np.ndarray] = None,
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    # _metric_protections(y, y_hat, weights)

    delta_y = np.square(y - y_hat)
    if weights is not None:
        mse = np.average(
            delta_y[~np.isnan(delta_y)], weights=weights[~np.isnan(delta_y)], axis=axis
        )
    else:
        mse = np.nanmean(delta_y, axis=axis)

    return mse


def mqloss(
    y: np.ndarray,
    y_hat: np.ndarray,
    quantiles: np.ndarray,
    weights: Optional[np.ndarray] = None,
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    if weights is None:
        weights = np.ones(y.shape)

    n_q = len(quantiles)

    y_rep = np.expand_dims(y, axis=-1)
    error = y_hat - y_rep
    sq = np.maximum(-error, np.zeros_like(error))
    s1_q = np.maximum(error, np.zeros_like(error))
    mqloss = quantiles * sq + (1 - quantiles) * s1_q

    # Match y/weights dimensions and compute weighted average
    weights = np.repeat(np.expand_dims(weights, axis=-1), repeats=n_q, axis=-1)
    mqloss = np.average(mqloss, weights=weights, axis=axis)

    return mqloss


def generate(
    output_directory,
    num_samples,
    ckpt_path,
    data_path,
    ckpt_iter,
    use_model,
    masking,
    missing_k,
    only_generate_missing,
):
    """
    Generate data based on ground truth

    Parameters:
    output_directory (str):           save generated speeches to this path
    num_samples (int):                number of samples to generate, default is 4
    ckpt_path (str):                  checkpoint path
    ckpt_iter (int or 'max'):         the pretrained checkpoint to be loaded;
                                      automitically selects the maximum iteration if 'max' is selected
    data_path (str):                  path to dataset, numpy array.
    use_model (int):                  0:DiffWave. 1:SSSDSA. 2:SSSDS4.
    masking (str):                    'mnr': missing not at random, 'bm': black-out, 'rm': random missing
    only_generate_missing (int):      0:all sample diffusion.  1:only apply diffusion to missing portions of the signal
    missing_k (int)                   k missing time points for each channel across the length.
    """
    seed_everything(9)
    # generate experiment (local) path
    local_path = "{}_{}_{}".format(
        args.data_path.split("/")[-1][:-4], args.pred_len, args.seed
    )

    # local_path = "T{}_beta0{}_betaT{}".format(diffusion_config["T"],
    #                                           diffusion_config["beta_0"],
    #                                           diffusion_config["beta_T"])

    # Get shared output_directory ready
    output_directory = os.path.join(output_directory, local_path)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    # predefine model
    if use_model == 0:
        net = DiffWaveImputer(**model_config).cuda()
    elif use_model == 1:
        net = SSSDSAImputer(**model_config).cuda()
    elif use_model == 2:
        net = SSSDS4Imputer(**model_config).cuda()
    else:
        print("Model chosen not available.")
    print_size(net)

    # load checkpoint
    ckpt_path = os.path.join(ckpt_path, local_path)
    if ckpt_iter == "max":
        ckpt_iter = find_max_epoch(ckpt_path)
    model_path = os.path.join(ckpt_path, "{}.pkl".format(ckpt_iter))
    print(model_path)
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        net.load_state_dict(checkpoint["model_state_dict"])
        print("Successfully loaded model at iteration {}".format(ckpt_iter))
    except:
        raise Exception("No valid model found")

    ### Custom data loading and reshaping ###
    _, testing_data = data_provider(args, flag="test")
    print("Data loaded")
    # testing_data = np.load(trainset_config['test_data_path'])
    # testing_data = np.split(testing_data, 4, 0)
    # testing_data = np.array(testing_data)
    # testing_data = torch.from_numpy(testing_data).float().cuda()

    # all_mse = []

    all_y_pred = []
    all_y_real = []
    for i, batch in enumerate(testing_data):
        batch = batch.cuda()
        if masking == "mnr":
            mask_T = get_mask_mnr(batch[0], missing_k)
            mask = mask_T.permute(1, 0)
            mask = mask.repeat(batch.size()[0], 1, 1)
            mask = mask.type(torch.float).cuda()

        elif masking == "bm":
            mask_T = get_mask_bm(batch[0], args.pred_len)
            mask = mask_T.permute(1, 0)
            mask = mask.repeat(batch.size()[0], 1, 1)
            mask = mask.type(torch.float).cuda()

        elif masking == "rm":
            mask_T = get_mask_rm(batch[0], missing_k)
            mask = mask_T.permute(1, 0)
            mask = mask.repeat(batch.size()[0], 1, 1).float().cuda()

        batch = batch.permute(0, 2, 1)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        sample_length = batch.size(2)
        sample_channels = batch.size(1)
        batch_size = batch.size(0)
        generated_audio = sampling(
            net,
            (num_samples * batch_size, sample_channels, sample_length),
            diffusion_hyperparams,
            cond=batch.repeat(num_samples, 1, 1),
            mask=mask.repeat(num_samples, 1, 1),
            only_generate_missing=only_generate_missing,
        )

        end.record()
        torch.cuda.synchronize()

        print(
            "generated {} utterances of random_digit at iteration {} in {} seconds".format(
                num_samples, ckpt_iter, round(start.elapsed_time(end) / 1000, 2)
            )
        )

        generated_audio = (
            generated_audio.permute(0, 2, 1)
            .reshape(batch_size, num_samples, sample_length, -1)
            .detach()
            .cpu()
            .numpy()
        )
        batch = batch.permute(0, 2, 1).detach().cpu().numpy()
        # mask = mask.permute(0,2,1).detach().cpu().numpy()
        all_y_pred.append(generated_audio[:, :, -args.pred_len :])
        all_y_real.append(batch[:, -args.pred_len :, :])

        # outfile = f'imputation{i}.npy'
        # new_out = os.path.join(ckpt_path, outfile)
        # np.save(new_out, generated_audio)

        # outfile = f'original{i}.npy'
        # new_out = os.path.join(ckpt_path, outfile)
        # np.save(new_out, batch)

        # outfile = f'mask{i}.npy'
        # new_out = os.path.join(ckpt_path, outfile)
        # np.save(new_out, mask)

        # print('saved generated samples at iteration %s' % ckpt_iter)
        # print(generated_audio.shape)
        # print(batch.shape)

        # mse = mean_squared_error(generated_audio[~mask.astype(bool)], batch[~mask.astype(bool)])
        # all_mse.append(mse)

    y_pred = np.concatenate(all_y_pred)
    y_real = np.concatenate(all_y_real)

    quantiles = (np.arange(9) + 1) / 10
    y_pred = y_pred.transpose((1, 0, 2, 3))
    y_pred_point = np.mean(y_pred, axis=0)
    y_pred_q = np.quantile(y_pred, quantiles, axis=0)
    y_pred_q = np.transpose(y_pred_q, (1, 2, 3, 0))

    MSE = mse(y_real, y_pred_point)
    CRPS = mqloss(y_real, y_pred_q, quantiles=np.array(quantiles))

    metric_out = os.path.join(ckpt_path, "metric.npy")
    np.save(metric_out, np.array([MSE, CRPS]))

    # print('Total MSE:', mean(all_mse))
    print(MSE, CRPS)
    # fig, ax = plt.subplots()
    # ax.plot(y_real[0].flatten())
    # ax.plot(y_pred_point[0].flatten())
    # fig.savefig('test.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config/config_SSSDS4.json",
        help="JSON file for configuration",
    )
    parser.add_argument("--data", type=str, default="custom")
    parser.add_argument(
        "--root_path", type=str, default="/home/user/data/THU-timeseries"
    )
    parser.add_argument("--data_path", type=str, default="electricity/electricity.csv")
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--label_len", type=int, default=0)
    parser.add_argument("--pred_len", type=int)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--features", type=str, default="S")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "-ckpt_iter",
        "--ckpt_iter",
        default="max",
        help='Which checkpoint to use; assign a number or "max"',
    )
    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        default=20,
        help="Number of utterances to be generated",
    )
    args = parser.parse_args()

    # Parse configs. Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    print(config)

    gen_config = config["gen_config"]

    train_config = config["train_config"]  # training parameters

    global trainset_config
    trainset_config = config["trainset_config"]  # to load trainset

    global diffusion_config
    diffusion_config = config["diffusion_config"]  # basic hyperparameters

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(
        **diffusion_config
    )  # dictionary of all diffusion hyperparameters

    global model_config
    if train_config["use_model"] == 0:
        model_config = config["wavenet_config"]
    elif train_config["use_model"] == 1:
        model_config = config["sashimi_config"]
    elif train_config["use_model"] == 2:
        model_config = config["wavenet_config"]
        model_config["s4_lmax"] = args.seq_len + args.pred_len

    generate(
        **gen_config,
        ckpt_iter=args.ckpt_iter,
        num_samples=args.num_samples,
        use_model=train_config["use_model"],
        data_path=trainset_config["test_data_path"],
        masking=train_config["masking"],
        missing_k=train_config["missing_k"],
        only_generate_missing=train_config["only_generate_missing"],
    )
