# Testing a conjecture for Metric Dimension of Grid + Edge

![](conjecture.png)

For a mxn grid augmented with one edge, to check that there is no resolving set of size 3 if and only if the MD=4 condition in our conjecture hold, run
```
python3 four_checker.py m n
```
To check that there is a resolving set of size 2 if and only if the MD=2 condition in our conjecture hold, run
```
python3 two_checker.py m n
```
Success message at the end of the output of the script indicates that the conjecture is verified.
Note that since we proved that the MD of the grid+edge is between 2 and 4, running both of these programs checks the entire statement of the conjecture.

Since we are doing an exhaustive brute-force search, running these files may take long time. For example, checking for 
9 x 9 grid when MD is 4 takes around 250 secs. To reduce this time, we have made an assumption that for every grid + edge, there 
will exist a resolving set with minimal size for which all the resolving points will lie on the boundary of the grid. This 
will reduce our search space for resolving set. We don't yet have a proof for this but it seems quite intuitive.
To verify with this assumption, run files ``opt_four_checker.py`` and ``opt_two_checker.py`` similarly.
This takes around 50 secs to verify 9 x 9 grid when MD is 4. 
