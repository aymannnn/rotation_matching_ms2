## How to Run the Algorithm

1. Most important: make sure your data matches the format of the test_data.csv. The column names are the most common error (usually sent a sheet with a space somewhere or the Name column named something else) and usually very easy to debug.
2. The more annoying error is if people reuse or number incorrectly, but lately hasn't been an issue.
3. Hard code the path variables to the real data and export in lines 19/20 in the python script (matching_algorithm_script.py)
4. That's it! Doesn't take long to run. 

## Notes for Future Updates/Implementation

1. If the rotation information number of available spots change you'l lhave to change that (the list in lines 72-88)
   a. Note that the most annoying part about this is that if there is some issue with availability in spots per block then it'll be very hard to debug.

## About the Algorithm

This serves to match students at Duke into their rotations, with the following contrains:

1. Two blocks, each block has restrictions on the amount of students per rotation (hard coded in)
2. Each student must have a general rotation and a subspecialty
3. Preferably one at DUH (for now ignored in favor of the above two)

Didn't use any linear optimization because a bit difficult to figure out with these constraints and blocks, so currently this is lazy greedy algorithm that essentially does the following:

1. Shuffle order of names
2. For each student, match into a general rotation, then a subspecialty
3. Repeat for each student
4. Calculate cost, defined as the sum of preference for each student -- goal is of course to minimize the cost

Then, repeat 1-4 about 10,000 times until a solution is found. This is a little more complex in reality because there are lots of conditions, checking to make sure there are spaces in general and subspeciality rotations in certain blocks before running the algorithm etc. But ultimately works real well. 

