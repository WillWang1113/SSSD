# from mega import Mega

# mega = Mega()
# m = mega.login()

# # Electricity
# print('Downloading Electricity dataset...')
# m.download_url('https://mega.nz/file/MaUxSTzI#SbVLiltqBzf9wa9C4zoE_goMYZY_rikTXFGodynSTmo', 'datasets/')
# m.download_url('https://mega.nz/file/lCElEBRA#RonhMz5aO4wDevhXTb6cqg_yjWegs0lRiY6yY2SuAvM', 'datasets/')
# print('Electricity dataset downloaded')

# # ETTm1
# print('Downloading ETTm1 dataset...')
# m.download_url('https://mega.nz/file/8LlFRKKA#1B15mP2Nu8bg2UXBvDvlb2wQtrhgqmr7NBgtgh9GynQ', 'datasets/')
# m.download_url('https://mega.nz/file/lbVHTKpJ#jIHxfQlx-WWW-tSX7juHsyYYyMrfEiQdGD17jvVINYk', 'datasets/')
# print('ETTm1 dataset downloaded')

# # Mujoco
# print('Downloading Mujoco dataset...')
# m.download_url('https://mega.nz/file/YKtwkSjZ#u_yJy3KNZyUfGYCFzwNNU7Nzar232-5r_jrCLSNmq50', 'datasets/')
# m.download_url('https://mega.nz/file/QXsBgQoS#CjbChp35YZ_sKIjrB0YLyuayfCWHSBuKJR2AZnZJO8k', 'datasets/')
# print('Mujoco dataset downloading')

# # PTB-XL
# print('Downloading PTB-XL dataset 248...')
# m.download_url('https://mega.nz/file/wKFmjbQY#9pQCnYAV282xJlkuJ1cAsgklHQj8toYCFylGZl5DC-w', 'datasets/')
# m.download_url('https://mega.nz/file/UDkByCLY#SwL3NyAhtkJKbvn6PEosnN9mTOZb4yT0PHaW6fMQU3k', 'datasets/')
# m.download_url('https://mega.nz/file/IOMzTIiI#w3wu0SNnelnDaoyn3cZXyvqTXLlf587SPWsyQOWYESc', 'datasets/')

# print('Downloading PTB-XL dataset 1000....')
# m.download_url('https://mega.nz/file/ZCtkFbZT#U4lDsYUZx_oiLX8QQVZLg4_bTFEM_xR2Xn3YCoTSPFM', 'datasets/')
# m.download_url('https://mega.nz/file/IS9Q1ZaT#3syB-EH_s0rI3riTzBnY9CtkGRxiUcBgqwsvig1uEQs', 'datasets/')
# m.download_url('https://mega.nz/file/oHUlUI4J#T4F3n1UdV0yZwZ1i9NZE_Vz-nHr5uxeYC49oh9jLqo4', 'datasets/')

# print('PTB-XL datasets downloaded')


import numpy as np
import os
import glob
import pandas as pd

seq_len = 96
pred_len = [96, 192, 336, 720]
data_dir = (
    "electricity",
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "exchange_rate",
    "traffic",
    "weather",
)
all_results = []

for d in data_dir:
    d_results = []
    for pl in pred_len:
        m_results = []
        
        exps = glob.glob(f"results/{d}_{pl}*")
        exps.sort()
        metrics = np.concatenate(
            [np.load(os.path.join(e, "metric.npy")).reshape(1, -1) for e in exps]
        )
        print(metrics)
            
        metrics = metrics.mean(axis=0)
        m_results.append(metrics)

        df = pd.DataFrame(m_results, columns=["MSE", "CRPS"])
        # df[["MSE", "CRPS"]] = df["metric"].tolist()
        # df = df.drop(columns="metric")
        df["pred_len"] = pl
        df["dataset"] = d
        d_results.append(df)
    d_results = pd.concat(d_results)
        
    all_results.append(d_results)
df = pd.concat(all_results)
print(df)
df.to_csv('results.csv')