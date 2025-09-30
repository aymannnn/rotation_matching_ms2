## About the Algorithm

This serves to match students at Duke into their rotations, with the following contrains:

1. Two blocks, each block has restrictions on the amount of students per rotation
2. Each student must have a general rotation and a subspecialty
3. Preferably one at DUH (for now ignored in favor of the above two), but this is currently not coded in. 

This is a lazy greedy algorithm that, in simple terms, does the following steps: 

1. Shuffle order of names
2. For each student, match into a general rotation, then a subspecialty
3. Repeat for each student
4. Calculate cost, defined as the sum of preference for each student -- goal is to minimize the cost (e.g. if you get your #1 preferences your cost is 2, if you get a #10 and a #2 it's 12)

Then, repeat 1-4 about many times until a solution is found.

## How to Run the Algorithm

1. Format your data! You may generate sample data with this script, but you have to make sure that your input data columns match EXACTLY ... I didn't implement a check, it will just fail if you don't.
2. I do check to make sure that people don't reuse or number incorrecty. If that's the case, their loss, you may do with that person's data what you wish.
3. Set the amount of iterations you would like to do on the slider. I recommend 10,000. More takes longer. 
4. Wait and download your final CSV!

## Modifying Rotation Capacity per Block

I have included an option on the side to modify the number of students per block. Be aware that the model conditions are strict - it will try and give each student a general rotation and subspeciality. If you reduce the number of general/subspecialty rotations it is possible that no solution will be possible with these constraints. It will not default to 2 general, or 2 subspecialty, etc.