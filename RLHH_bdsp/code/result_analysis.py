import numpy as np
import pandas as pd


def concat_rlhh(RLHH_file, baseline_filename):
	h = pd.read_csv(f"../result/{baseline_filename}.csv")
	if '.csv' in RLHH_file:
		hh = pd.read_csv(RLHH_file)
	else:
		hh = pd.read_csv(f"../result/{RLHH_file}.csv")
	hh['method'] = "RLHH"
	df = pd.concat([h, hh])
	df = df.sort_values(by=['No.', 'method'], ascending=True)
	df['objval'] = df['objval'].apply(lambda x: round(x))
	df['time'] = df['time'].apply(lambda x: round(x, 2))
	return df

def count(df):
	df_count = df[['No.', 'n', 'method']].groupby(['n', 'method'], as_index=False).count()
	df_count = df_count.pivot('n', 'method', 'No.')
	df_count.fillna(0, inplace=True)

	df_count['sum'] = df_count.sum(axis=1)
	df_count.loc['sum'] = df_count.sum(axis=0)
	df_count = df_count.astype(int)

	return df_count

def analysis(df, summary_dir=None):
	# 先比最优值，再比速度
	df_best_fast = df.sort_values(by=['objval', 'time'], ascending=True)
	df_best_fast = df_best_fast.groupby(['No.']).head(1)
	best_fast_count = count(df_best_fast)
	print(best_fast_count)
	print(best_fast_count.to_latex())
	print(best_fast_count.T.to_latex())

	# 最优值比较
	df_best = df.groupby('No.').apply(lambda t: t[t.objval==t.objval.min()]).reset_index(drop=True)
	best_count = count(df_best)
	best_count['sum'] = best_fast_count['sum']
	print(best_count)
	print(best_count.to_latex())
	print(best_count.T.to_latex())

	# 速度比较
	df_fast = df.groupby('No.').apply(lambda t: t[t.time == t.time.min()]).reset_index(drop=True)
	fast_count = count(df_fast)
	# print(fast_count)

	if summary_dir is not None:
		with pd.ExcelWriter(summary_dir) as writer:
			best_fast_count.to_excel(writer, sheet_name="best_fast")
			best_count.to_excel(writer, sheet_name="best")
			fast_count.to_excel(writer, sheet_name="fast")

	if 'RLHH' in best_count.columns:
		count1 = best_fast_count['RLHH'].values.tolist()
		count2 = best_count['RLHH'].values.tolist()
		return count1 + count2
	else:
		return None

def main(RLHH_file=None, baseline_filename="base_50-100(600s)"):
	if RLHH_file is None:
		df = pd.read_csv(f"../result/{baseline_filename}.csv")
		return analysis(df, summary_dir="../result/summary/baseline.xlsx")
	else:
		df = concat_rlhh(RLHH_file, baseline_filename)
		df.to_csv(f"../result/detail/{RLHH_file}.csv", index=False)
		return analysis(df, summary_dir=f"../result/summary/{RLHH_file}.xlsx")


if __name__ == "__main__":
	# main("small_0.0_a=100_e=0.05_g=0.95_lr=0.1_seed=123_19.56-11.23")
	main()	# for debug
