# %%
from os import sep
import pandas as pd
import polars as pl

DATA_PATH = "../data/popularity-contest"

# %%
# Parsing Unix timestamps
# It's not obvious how to deal with Unix timestamps in pandas -- it took me quite a while to figure this out. The file we're using here is a popularity-contest file of packages.

# Read it, and remove the last row
popcon = pd.read_csv(
    "../data/popularity-contest",
    sep=" ",
)[:-1]
popcon.columns = ["atime", "ctime", "package-name", "mru-program", "tag"]
popcon[:5]

# TODO: please reimplement this using Polars
pl_popcon = pl.read_csv(DATA_PATH, separator=" ", infer_schema=0)[:-1]
# NOTE: or you can do this: 
nr_rows = (
    pl.scan_csv(DATA_PATH, separator=" ", infer_schema=0)
    .select(pl.len())
    .collect()
    .item()
)
pl_popcon = pl.read_csv(DATA_PATH, separator=" ", infer_schema=0, n_rows=(nr_rows - 1))

pl_popcon.columns = ["atime", "ctime", "package-name", "mru-program", "tag"]
pl_popcon.head(5)

# %%
# The magical part about parsing timestamps in pandas is that numpy datetimes are already stored as Unix timestamps. So all we need to do is tell pandas that these integers are actually datetimes -- it doesn't need to do any conversion at all.
# We need to convert these to ints to start:
popcon["atime"] = popcon["atime"].astype(int)
popcon["ctime"] = popcon["ctime"].astype(int)

# TODO: please reimplement this using Polars
pl_popcon = pl_popcon.with_columns(
    pl.col("atime").cast(pl.Int64),
    pl.col("ctime").cast(pl.Int64),
)

# %%
# Every numpy array and pandas series has a dtype -- this is usually `int64`, `float64`, or `object`. Some of the time types available are `datetime64[s]`, `datetime64[ms]`, and `datetime64[us]`. There are also `timedelta` types, similarly.
# We can use the `pd.to_datetime` function to convert our integer timestamps into datetimes. This is a constant-time operation -- we're not actually changing any of the data, just how pandas thinks about it.

popcon["atime"] = pd.to_datetime(popcon["atime"], unit="s")
popcon["ctime"] = pd.to_datetime(popcon["ctime"], unit="s")
popcon.head()

# TODO: please reimplement this using Polars
# convert to milliseconds
pl_popcon = pl_popcon.with_columns(
    pl.col("atime") * 1000,
    pl.col("ctime") * 1000,
)
pl_popcon = pl_popcon.with_columns(
    pl.col("atime").cast(pl.Datetime("ms")),
    pl.col("ctime").cast(pl.Datetime("ms")),
)

# %%
# Now suppose we want to look at all packages that aren't libraries.

# First, I want to get rid of everything with timestamp 0. Notice how we can just use a string in this comparison, even though it's actually a timestamp on the inside? That is because pandas is amazing.
popcon = popcon[popcon["atime"] > "1970-01-01"]

# Now we can use pandas' magical string abilities to just look at rows where the package name doesn't contain 'lib'.
nonlibraries = popcon[~popcon["package-name"].str.contains("lib")]
nonlibraries.sort_values("ctime", ascending=False)[:10]

# TODO: please reimplement this using Polars
pl_popcon = pl_popcon.filter(pl.col("atime") > pl.datetime(year=1970, month=1, day=1))
pl_nonlibraries = pl_popcon.filter(~pl.col("package-name").str.contains("lib"))

# The whole message here is that if you have a timestamp in seconds or milliseconds or nanoseconds, then you can just "cast" it to a `'datetime64[the-right-thing]'` and pandas/numpy will take care of the rest.
