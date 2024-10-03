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

Hard code the path variables to the real data and export. That's it!

TODO:

One day will maybe implement a small check, where if somebody matches into their top 1-3 or so pick for general then will just randomly shuffle the subspecialties instead of ranking them too. 