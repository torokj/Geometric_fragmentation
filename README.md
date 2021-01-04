# Geometric_fragmentation
Geometric fragmentation of rocks
randomcut1.py: cuts a cube by random planes. Prints the resulting polyhedra in a special format
SU_break_light_multi_var_face.py: further breaks the resulting fragments of randomcut1.py,
based on certain criteria, and calculates a number of geometric descriptors

usage: randomcut1.py N seed outtype(0:rep, 1:dat)
N: number of planes
seed: seed for the random number generator. Supplying the same number results in an identical result
outtype=0: output to create 3d visualization with http://www.phy.bme.hu/~torok/Mozi/
outtype=1: data output. Data format: each fragment is marked with a line consisting the single word "object". Example:

object
vertices 6
0.885450273711 0.22777276281 0.130279162742
0.802469222196 0.487851436775 0.0302212127772
0.974163868997 0.305857790183 0.237853035495
0.592336300068 0.356583246652 0.343141049306
0.661249737978 0.564721461343 0.12215766833
0.613632922422 0.485824170269 0.53130400486
faces 5
1 0 2
1 0 3 4
1 4 5 2
2 5 3 0
4 5 3

This fragment has 6 vertices as indicated by the second line. The next six lines contain the positions of the vertices. The fragment has 5 faces and the next five lines detail the corners of the faces. The numbers reference the vertices in the given order starting from 0
