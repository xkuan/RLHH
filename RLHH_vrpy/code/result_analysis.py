import numpy as np
import pandas as pd


def concat_rlhh(RLHH_file, baseline_filename):
	h = pd.read_csv(f"../result/baseline/{baseline_filename}.csv")
	if '.csv' in RLHH_file:
		hh = pd.read_csv(RLHH_file)
	else:
		hh = pd.read_csv(f"../result/{RLHH_file}.csv")
	hh['method'] = "RLHH"
	# 只比较 RLHH 中有的案例
	h = pd.merge(h, hh[['instance', 'n']], how='inner', on=['instance', 'n'])
	hh = pd.merge(hh, h[['instance', 'n']].drop_duplicates(), how='inner', on=['instance', 'n'])

	df = pd.concat([h, hh])
	df = df[['No.', 'type', 'instance', 'n', 'method', 'iters', 'objval', 'time']]
	if "type" not in df.columns or df["type"].isna().any():
		df["type"] = df['instance'].apply(lambda x: x[:-2])
	df = df.sort_values(by=['n', 'instance', 'method'], ascending=True)
	df['objval'] = df['objval'].apply(lambda x: round(x, 1))
	df['time'] = df['time'].apply(lambda x: round(x, 2))
	return df

def count(df, guroup_by):
	df_count = df[['instance', 'method', guroup_by]].groupby([guroup_by, 'method'], as_index=False).count()
	df_count = df_count.pivot(guroup_by, 'method', 'instance')
	df_count.fillna(0, inplace=True)

	df_count['sum'] = df_count.sum(axis=1)
	df_count.loc['sum'] = df_count.sum(axis=0)
	df_count = df_count.astype(int)

	return df_count

def analysis(df, summary_dir=None):
	# 先比最优值，再比速度
	df_best_fast = df.sort_values(by=['objval', 'time'], ascending=True)
	df_best_fast = df_best_fast.groupby(['instance', 'n']).head(1)
	best_fast_count = count(df_best_fast, guroup_by='type')
	print(best_fast_count)
	print(best_fast_count.to_latex())
	print(best_fast_count.T.to_latex())

	# 最优值比较
	df_best = df.groupby(['instance', 'n']).apply(lambda t: t[t.objval==t.objval.min()]).reset_index(drop=True)
	best_count = count(df_best, guroup_by='type')
	best_count['sum'] = best_fast_count['sum']
	print(best_count)
	print(best_count.to_latex())
	print(best_count.T.to_latex())

	# 速度比较
	df_fast = df.groupby(['instance', 'n']).apply(lambda t: t[t.time == t.time.min()]).reset_index(drop=True)
	fast_count = count(df_fast, guroup_by='type')
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

def main(RLHH_file=None, baseline_filename="base_25-35"):
	if RLHH_file is None:
		df = pd.read_csv(f"../result/baseline/{baseline_filename}.csv")
		return analysis(df, summary_dir="../result/summary/baseline.xlsx")
	else:
		df = concat_rlhh(RLHH_file, baseline_filename)
		df.to_csv(f"../result/detail/{RLHH_file}.csv", index=False)
		return analysis(df, summary_dir=f"../result/summary/{RLHH_file}.xlsx")


if __name__ == "__main__":
	main("large_r_a=100_e=0.05_g=0.95_lr=0.01_seed=523_17.19-11.04", baseline_filename="large_instances_to_r202_100")
	# main()
