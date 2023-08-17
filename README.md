# duckdb-vector
A duckdb extension to add support for vectors([not the duckdb internal representation](https://duckdb.org/internals/vector.html#vector-format)).

## Vector Distance
This extension adds support for vector distances using the function `list_distance`.
The third argument to this function is the distance algorithm to be used.
Note: The distance algorithm should be a binary aggregate.
Eg:
```sql
SELECT list_distance([1, 1, 1], [1, 1, 3], 'dot_product') FROM vectors;
----
5
```
You can also use the macro for this which is of the form `list_<distance_algorithm>`.

The following distance algorithms are supported: 
1. `l2distance` or `euclidean_distance`: $\sqrt{\sum_{i=1}^n {(y_i - x_i)}^2}$
2. `dot_product`: $\sum_{i=1}^{n}x_iy_i$
3. `cosine_similarity`: $\frac{\sum_{i=1}^{n}x_ib_i}{\sqrt{{\sum_{i=1}^{n}{x_i}^2}{\sum_{i=1}^{n}{y_i}^2}}}$
4. `cosine_distance`: $1 - cosineSimilarity$ or $1 - \frac{\sum_{i=1}^{n}x_ib_i}{\sqrt{{\sum_{i=1}^{n}{x_i}^2}{\sum_{i=1}^{n}{y_i}^2}}}$
5. `l2norm`: $\sqrt{\sum_{i=1}^n {x_i}^2}$ (Since this is a unary aggregate function, it can be used with `list_aggr` or `list_l2norm`).
