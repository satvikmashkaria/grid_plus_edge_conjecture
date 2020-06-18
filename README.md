# Testing a conjecture for Metric Dimension of Grid + edge

TODO - put conjecture here

To verify this conjecture for a m x n grid when MD is 4, run
```
python3 four_checker.py m n
```
If Success message is printed at the end, conjecture is verified.
Similarly, for verifying conjecture when MD is 2, run
```
python3 two_checker.py m n
```

Since we are doing an exhaustive brute-force search, running these files may take long time. For example, checking for 
9 x 9 grid when MD is 4 takes around 250 secs. To reduce this time, we have made an assumption that for every grid + edge, there 
will exist a resolving set with minimal size for which all the resolving points will lie on the boundary of the grid. This 
will reduce our search space for resolving set. We don't yet have a proof for this but it seems quite intuitive.
To verify with this assumption, run files ``opt_four_checker.py`` and ``opt_two_checker.py`` similarly.
This takes around 50 secs to verify 9 x 9 grid when MD is 4. 
