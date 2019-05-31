
import numpy as np
import pandas as pd

# 上半年
df_2017_1 = pd.read_csv("./data/2017/1_up")
df_2017_3 = pd.read_csv("./data/2017/3_up")
df_2017_5 = pd.read_csv("./data/2017/5_up")
df_2017_6 = pd.read_csv("./data/2017/6_up")

df_2017_3[6] = df_2017_5[3]
df_2017_3[7] = df_2017_6[3]
df_2017_3[8] = df_2017_1[3]

df_2017_3.columns = ["date", "Generator_speed", "Rotor_speed", "Gearbox_oil_tem",
                     "Generator_bearing_tem_drive", "Generator_bearing_tem_nondrive",
                     "wind_speed", "active_power", "state"]
print(df_2017_3.head(20))