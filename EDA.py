#%%
import os
import time
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
#%%
DATA_DIR = "Data"
TRAIN_PATH = os.path.join(DATA_DIR, "train-dset.parquet")
TEST_PATH = os.path.join(DATA_DIR, "test-dset-small.parquet")
#%% md
# # 1. Размеры
#%%
t0 = time.time()

train_lf = pl.scan_parquet(TRAIN_PATH)
test_lf = pl.scan_parquet(TEST_PATH)


train = train_lf.collect(streaming=True)
test = test_lf.collect(streaming=True)

print(f"[load] train: {train.shape}, test: {test.shape}, time: {time.time()-t0:.2f}s\n")
#%% md
# # 2. Cхема
#%%
print("[schema] dtypes (train):")
for k, v in train.schema.items():
    print(f"  {k}: {v}")
print()
#%%
train.head(5)
#%% md
# # 3. Пропуски
#%%
missing = (
    train
    .select([pl.col(c).is_null().sum().alias(c) for c in train.columns])
    .transpose(include_header=True, header_name="column", column_names=["nulls"])
    .with_columns([
        (pl.col("nulls") / train.height).alias("null_frac")
    ])
    .sort("nulls", descending=True)
)
print("[missing] Количество и доля пропусков по колонкам:")
missing

#%%
print(missing)
#%% md
# # 4. Уникальные значения
#%%
def nunique(df: pl.DataFrame, col: str) -> int:
    return df.select(pl.col(col).n_unique()).item()

print("[unique] уникальных query_id (train/test):", nunique(train, "query_id"), nunique(test, "query_id"))
print("[unique] уникальных item_id   (train/test):", nunique(train, "item_id"), nunique(test, "item_id"))
print()
#%% md
# # Числовые данные
#%%
num_cols = [c for c, t in train.schema.items() if t in pl.NUMERIC_DTYPES]
if num_cols:
    # Для компактности считаем набор статистик по каждой числовой колонке
    stats_rows = []
    for c in num_cols:
        s = (
            train
            .select(
                pl.col(c).cast(pl.Float64),
                pl.len().alias("__n")
            )
            .select([
                pl.lit(c).alias("column"),
                pl.col(c).mean().alias("mean"),
                pl.col(c).std().alias("std"),
                pl.col(c).min().alias("min"),
                pl.col(c).quantile(0.25).alias("p25"),
                pl.col(c).median().alias("p50"),
                pl.col(c).quantile(0.75).alias("p75"),
                pl.col(c).quantile(0.95).alias("p95"),
                pl.col(c).quantile(0.99).alias("p99"),
                pl.col(c).max().alias("max"),
            ])
        )
        stats_rows.append(s)
    num_stats = pl.concat(stats_rows)
    print("[numeric] базовые статистики:")
    res = num_stats.sort("column")
#%%
res
#%%
print(res)
#%% md
# # Таргет: баланс классов
#%%
if "item_contact" in train.columns:
    tgt = (
        train
        .with_columns(pl.col("item_contact").cast(pl.Int32))
        .group_by("item_contact")
        .agg(pl.len().alias("cnt"))
        .with_columns((pl.col("cnt") / pl.col("cnt").sum()).alias("frac"))
        .sort("item_contact")
    )
    print("[target] распределение item_contact:")
    print(f"[target] положительная доля: {train.select(pl.col('item_contact').cast(pl.Float64).mean()).item():.4f}\n")
#%%
tgt
#%%
print(tgt)
#%% md
# # Группы по запросам: размер пула, доля позитивов, цена внутри запроса
#%%
if "item_contact" in train.columns and "price" in train.columns:
    per_query = (
        train
        .with_columns(pl.col("item_contact").cast(pl.Int32))
        .group_by("query_id")
        .agg([
            pl.len().alias("n_items"),
            pl.col("item_contact").sum().alias("n_pos"),
            pl.col("item_contact").mean().alias("pos_rate"),
            pl.col("price").mean().alias("price_mean"),
            pl.col("price").median().alias("price_median")
        ])
        .sort("n_items", descending=True)
    )
    print("[group] примеры агрегатов по query_id:")
    print(per_query.head(10))
    print()

    # Распределение размера пулов и доли позитивов (в числах)
    print("[group] размер пула кандидатов (квантили):")
    print(per_query.select([
        pl.col("n_items").min().alias("min"),
        pl.col("n_items").quantile(0.5).alias("p50"),
        pl.col("n_items").quantile(0.9).alias("p90"),
        pl.col("n_items").quantile(0.99).alias("p99"),
        pl.col("n_items").max().alias("max"),
    ]))
    print()
#%% md
#  # Категории и локации
#%%
for a, b, name in [
    ("query_cat", "item_cat_id", "cat_match"),
    ("query_mcat", "item_mcat_id", "mcat_match"),
    ("query_loc", "item_loc", "loc_match"),
]:
    if a in train.columns and b in train.columns:
        rate = (
            train
            .with_columns((pl.col(a) == pl.col(b)).alias(name))
            .select(pl.col(name).cast(pl.Int32).mean())
            .item()
        )
        print(f"[match] доля совпадений {a} == {b}: {rate:.4f}")
print()
#%%
def topk(df: pl.DataFrame, col: str, k: int = 10) -> pl.DataFrame:
    return df.group_by(col).agg(pl.len().alias("cnt")).sort("cnt", descending=True).head(k)

for col in ["query_cat", "query_mcat", "query_loc", "item_cat_id", "item_mcat_id", "item_loc"]:
    if col in train.columns:
        print(f"[top] {col}:")
        print(topk(train, col, 10))
        print()
#%% md
# # Тексты
#%%
text_cols = [c for c, t in train.schema.items() if t == pl.Utf8]
has_text = all(x in train.columns for x in ["query_text", "item_title", "item_description"])

if has_text:
    # Предобработка: нижний регистр, очистка, токенизация по пробелам
    # Регулярка удаляет все не-буквенно-цифровые символы, сводит последовательности к одному пробелу.
    def tokenize(col: str) -> pl.Expr:
        return (
            pl.col(col)
            .cast(pl.Utf8)
            .fill_null("")
            .str.to_lowercase()
            .str.replace_all(r"[^0-9\p{L}]+", " ")
            .str.strip_chars()  # <--- замена .strip()
            .str.split(" ")
            .list.eval(pl.element().filter(pl.element() != ""))  # убираем пустые
        )


    train = train.with_columns([
        tokenize("query_text").alias("query_tokens"),
        tokenize("item_title").alias("title_tokens"),
        tokenize("item_description").alias("desc_tokens"),
    ])

    # Длины в словах
    train = train.with_columns([
        pl.col("query_tokens").list.len().alias("query_len"),
        pl.col("title_tokens").list.len().alias("title_len"),
        pl.col("desc_tokens").list.len().alias("desc_len"),
    ])

    print("[text] квантили длин (слов):")
    print(
        train.select([
            pl.col("query_len").quantile(0.5).alias("query_p50"),
            pl.col("query_len").quantile(0.95).alias("query_p95"),
            pl.col("title_len").quantile(0.5).alias("title_p50"),
            pl.col("title_len").quantile(0.95).alias("title_p95"),
            pl.col("desc_len").quantile(0.5).alias("desc_p50"),
            pl.col("desc_len").quantile(0.95).alias("desc_p95"),
        ])
    )
    print()

    # Пересечение токенов между запросом и заголовком/описанием
    # Используем set_intersection для уникальных токенов
    train = train.with_columns([
        pl.col("query_tokens").list.set_intersection(pl.col("title_tokens")).list.len().alias("overlap_q_title"),
        pl.col("query_tokens").list.set_intersection(pl.col("desc_tokens")).list.len().alias("overlap_q_desc"),
    ])

    print("[text] среднее пересечение токенов:")
    print(
        train.select([
            pl.col("overlap_q_title").mean().alias("mean_overlap_q_title"),
            pl.col("overlap_q_desc").mean().alias("mean_overlap_q_desc"),
        ])
    )
    print()

#%% md
# # Поведенческий сигнал vs таргет
#%%
if "item_query_click_conv" in train.columns and "item_contact" in train.columns:
    # Корреляция Пирсона (item_contact приводим к float)
    corr = train.select(pl.corr(pl.col("item_query_click_conv"), pl.col("item_contact").cast(pl.Float64))).item()
    print(f"[corr] corr(item_query_click_conv, item_contact): {corr:.4f}")

    # Средние значения для 0/1
    by_target = (
        train
        .with_columns(pl.col("item_contact").cast(pl.Int32))
        .group_by("item_contact")
        .agg(pl.col("item_query_click_conv").mean().alias("conv_mean"))
        .sort("item_contact")
    )
    print("[conv] среднее conv по классам item_contact:")
    print(by_target)
    print()

#%%
def sample_for_plots(df: pl.DataFrame, n: int = 200_000, seed: int = 42) -> pl.DataFrame:
    if df.height <= n:
        return df
    return df.sample(n=n, shuffle=True, seed=seed)
#%%
plot_df = sample_for_plots(train)

# Гистограмма цен (лог-шкала)
if "price" in plot_df.columns:
    price_vals = plot_df.select(pl.col("price").cast(pl.Float64)).to_series().to_numpy()
    price_vals = price_vals[np.isfinite(price_vals) & (price_vals > 0)]
    if len(price_vals) > 0:
        plt.figure(figsize=(6, 4))
        plt.hist(price_vals, bins=100)
        plt.xscale("log")
        plt.title("Распределение цен (лог-шкала)")
        plt.xlabel("price")
        plt.ylabel("count")
        plt.tight_layout()
        plt.show()
#%%
# Гистограмма item_query_click_conv
if "item_query_click_conv" in plot_df.columns:
    conv_vals = plot_df.select(pl.col("item_query_click_conv").cast(pl.Float64)).to_series().to_numpy()
    conv_vals = conv_vals[np.isfinite(conv_vals)]
    if len(conv_vals) > 0:
        plt.figure(figsize=(6, 4))
        plt.hist(conv_vals, bins=80)
        plt.title("Распределение item_query_click_conv")
        plt.xlabel("item_query_click_conv")
        plt.ylabel("count")
        plt.tight_layout()
        plt.show()

#%%
# Длины текстов
if has_text:
    q_len = plot_df.select(pl.col("query_len").cast(pl.Float64)).to_series().to_numpy()
    t_len = plot_df.select(pl.col("title_len").cast(pl.Float64)).to_series().to_numpy()
    d_len = plot_df.select(pl.col("desc_len").cast(pl.Float64)).to_series().to_numpy()

    plt.figure(figsize=(6, 4))
    plt.hist(q_len[np.isfinite(q_len)], bins=50)
    plt.title("Длина запроса (слов)")
    plt.xlabel("query_len")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.hist(t_len[np.isfinite(t_len)], bins=50)
    plt.title("Длина заголовка (слов)")
    plt.xlabel("title_len")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.hist(d_len[np.isfinite(d_len)], bins=50)
    plt.title("Длина описания (слов)")
    plt.xlabel("desc_len")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()
#%%
test.head(5)
#%%
ex = train.filter(pl.col("query_id") == 757116)
ex
#%%
