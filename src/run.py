import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# RUN TRAINING


seq_len = 96
pred_len = [96, 192, 336, 720]
data_dir = [
    # "electricity/electricity.csv",
    # "ETT-small/ETTh1.csv",
    "ETT-small/ETTh2.csv",
    # "ETT-small/ETTm1.csv",
    "ETT-small/ETTm2.csv",
    # "exchange_rate/exchange_rate.csv",
    # "traffic/traffic.csv",
    # "weather/weather.csv",
]

for i in range(5):
    for pl in pred_len:
        for dt in data_dir:
            if dt.split("/")[-1][:-4] not in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
                data_class = "custom"
            else:
                data_class = dt.split("/")[-1][:-4]

            
            os.system(
                f"python train.py --data_path {dt} --pred_len {pl} --seq_len {seq_len} --seed {i} --data {data_class}"
            )
            os.system(
                f"python inference.py --data_path {dt} --pred_len {pl} --seq_len {seq_len} --seed {i} --data {data_class}"
            )
 


# seq_len = 288
# pred_len = [288, 432, 576]
# data_dir = "MFRED/MFRED.csv"
# for i in range(5):
#     for pl in pred_len:
#         os.system(
#             f"python train.py --data_path {data_dir} --pred_len {pl} --seq_len {seq_len} --seed {i}"
#         )
#         os.system(
#             f"python inference.py --data_path {data_dir} --pred_len {pl} --seq_len {seq_len} --seed {i}"
#         )

# # COLLECT METRICS
# all_df = []
# for dt, d in data_dir:
#     df = []
#     for pl in pred_len:
#         print(dt, pl)
#         # TODO: 5 runs
#         seed_results = []
#         for ii in range(5):
#             result_pth = os.path.join(f'save/forecasting_{dt}_{ii}_{pl}', 'result_nsample100.pk')
#             with open(result_pth, 'rb') as f:
#                 results = np.array(pickle.load(f))
#             seed_results.append(results.reshape(1,-1))
#         seed_results = np.concatenate(seed_results)
#         print(seed_results)
#         df.append(seed_results.mean(axis=0, keepdims=True))
#     df = np.concatenate(df)
#     df = pd.DataFrame(df, columns=["MSE", "CRPS"])
#     df['dataset'] = dt
#     df['pred_len'] = pred_len
#     all_df.append(df)
# all_df = pd.concat(all_df)
# all_df = all_df[["dataset", "pred_len", "MSE", 'CRPS']]
# all_df.to_csv('mfred_csdi_forecast_result.csv')

# result_pth = 'save/forecasting_mfred_0_288/generated_outputs_nsample100.pk'
# with open(result_pth, 'rb') as f:
#     results = pickle.load(f)

# all_generated_samples = results[0]
# all_target = results[1]

# pred_len = 288

# quantiles=(np.arange(9) + 1) / 10
# y_pred = all_generated_samples.transpose((1,0,2,3))
# y_pred_point = np.mean(y_pred, axis=0)[:, -pred_len:, :]
# y_pred_q = np.quantile(y_pred, quantiles, axis=0)
# y_pred_q = np.transpose(y_pred_q, (1,2,3,0))[:, -pred_len:, :, :]
# y_real = all_target[:, -pred_len:, :]

# fig, ax = plt.subplots()
# ax.plot(y_real[0])
# ax.plot(y_pred_point[0])
# ax.plot(y_pred_q[0,:,0])
# fig.savefig('test.png')
