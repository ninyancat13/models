# FAQ

### Seaborn vs Matplotlib?
Seaborn is a high-level library. It provides simple codes to visualize complex statistical plots, which also happen to be aesthetically pleasing. But Seaborn was built on top of Matplotlib, meaning it can be further powered up with Matplotlib functionalities. Best way is to learn Seaborn first, spend some time visualizing with it, understand its limitations. Seaborn’s limitations will naturally lead you into Matplotlib.

https://towardsdatascience.com/seaborn-can-do-the-job-then-why-matplotlib-dac8d2d24a5f

### Why do we use pandas loc when we already have something like df['time']?

- Explicit is better than implicit.
df[boolean_mask] selects rows where boolean_mask is True, but there is a corner case when you might not want it to: when df has boolean-valued column labels. Thus, df[boolean_mask] does not always behave the same as df.loc[boolean_mask]. Even though this is arguably an unlikely use case, I would recommend always using df.loc[boolean_mask] instead of df[boolean_mask] because the meaning of df.loc's syntax is explicit. With df.loc[indexer] you know automatically that df.loc is selecting rows. In contrast, it is not clear if df[indexer] will select rows or columns (or raise ValueError) without knowing details about indexer and df.


- df.loc[row_indexer, column_index] can select rows and columns. df[indexer] can only select rows or columns depending on the type of values in indexer and the type of column values df has (again, are they boolean?).
```
In [237]: df2.loc[[True,False,True], 'B']
Out[237]: 
0    3
2    5
Name: B, dtype: int64
```

- When a slice is passed to df.loc the end-points are included in the range. When a slice is passed to df[...], the slice is interpreted as a half-open interval:
```
In [239]: df2.loc[1:2]
Out[239]: 
   A  B
1  2  4
2  3  5

In [271]: df2[1:2]
Out[271]: 
   A  B
1  2  4
```

https://stackoverflow.com/questions/38886080/python-pandas-series-why-use-loc

### Difference between loc and iloc and combining the two together?
The main distinction between the two methods is:
- loc gets rows (and/or columns) with particular labels.
- iloc gets rows (and/or columns) at integer locations.

loc's label-querying capabilities extend well-beyond integer indexes and it's worth highlighting a couple of additional examples.

Combining iloc and loc (aka when we want to find data via integer position and label)
```
df.iloc[:df.index.get_loc('c') + 1, :4]
```

https://stackoverflow.com/questions/31593201/how-are-iloc-and-loc-different/31593712#31593712


### Why pandas python syntax can seem inconsistent (compared to a similar package such as dplr in R)?
It is because pandas needs to conform to Python's existing syntax rules, which are pretty strict with respect to what unquoted symbols can represent (basically objects in the current scope). 

https://stackoverflow.com/questions/44060100/seemingly-inconsistent-column-reference-syntax-when-chaining-methods-on-pandas-d
