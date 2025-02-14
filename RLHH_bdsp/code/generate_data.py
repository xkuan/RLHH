import numpy as np
import pandas as pd
import random
import datetime


def generate_data(shift_num):
    start_hour_pdf = [3, 3, 5, 9, 10, 8, 5, 4, 3, 3, 4, 5, 9, 10, 8, 5, 3, 3]
    # shift_id = np.arange(1, shift_num+1)
    start_hour = np.array(random.choices(list(range(4, 22)), weights=start_hour_pdf, k=shift_num))
    start_minute = np.random.randint(0, 59, size=shift_num)
    duration = np.random.normal(loc=55 , scale=20, size=shift_num)
    duration = np.array([max(20, min(90, int(i))) for i in duration])

    start_time_str = np.array([datetime.time(hour, minute).strftime("%H:%M") for hour, minute in zip(start_hour, start_minute)])
    start_time = np.array([datetime.datetime(2022, 11, 18, hour, minute) for hour, minute in zip(start_hour, start_minute)])
    end_time_str = np.array([(start_time[i] + datetime.timedelta(minutes=int(duration[i]))).strftime("%H:%M") for i in range(shift_num)])

    start_minute = np.array([hour * 60 + minute for hour, minute in zip(start_hour, start_minute)])
    end_minute = np.array([start + spread for start, spread in zip(start_minute, duration)])

    df = pd.DataFrame(
        np.concatenate([start_time_str.reshape(shift_num,1),
        end_time_str.reshape(shift_num,1),
        start_minute.reshape(shift_num,1),
        end_minute.reshape(shift_num,1),
        duration.reshape(shift_num,1)], axis=1),
        columns=["start_time", "end_time", "start_minute", "end_minute", "duration"]
    )
    df[["start_minute", "end_minute", "duration"]] = df[["start_minute", "end_minute", "duration"]].apply(pd.to_numeric)
    df = df.sort_values(by="start_minute", ascending=True).reset_index(drop=True)
    # df.to_csv("../data/wuhu.csv", index_label="shift_id")

    return df


if __name__ == "__main__":
    random.seed(123)
    scales = [50, 75, 100, 150, 200]
    prob_num = 30
    for scale in scales:
        for number in range(prob_num):
            data = generate_data(shift_num=scale)
            data.to_csv("../data/shift_{}_{:02d}.csv".format(scale, number+1), index=False)
