#!/usr/bin/python

import math
import sys
import os
import numpy as np

def pointsonplane(NP,s,i):
  rv = []
  for a in range(NP):
    if s[a][0] == i or s[a][1] == i or s[a][2] == i:
      rv.append(a)
  return rv

def neigh_plane(nodes,nodesonplane):
  nop = []
  for n in nodes:
    if n in nodesonplane:
      nop.append(n)
  return nop

def loop(cs1,cs2,cs3,nodes,p,nodesonplane):
  #print "#START",cs1,cs2,cs3,nodesonplane,p[cs1],p[cs2],p[cs3]
  cslist = [ cs1, cs2, cs3]
  norm = np.cross(p[cs1] - p[cs2],p[cs3] - p[cs2])
  while cs3 != cs1:
    found = 0
    for cs in neigh_plane(nodes[cs3],nodesonplane):
      tmpv = np.cross(p[cs2]-p[cs3], p[cs]-p[cs3])
      tmpf = np.dot(p[cs]-p[cs3],p[cs2]-p[cs3])
      tmpf /= np.linalg.norm(p[cs2]-p[cs3])
      tmpf /= np.linalg.norm(p[cs]-p[cs3])
      if np.dot(tmpv,norm) > 1e-10 and tmpf < 0.99999999:
	cs2 = cs3
	cs3 = cs
	if cs != cs1:
	  cslist.append(cs)
	found = 1
        break
    if not found:
      break
    #print cs1,cs2,cs3,p[cs1],p[cs2],p[cs3],np.dot(tmpv,norm),tmpf
  #print "FOUND",found,cslist
  if found:
    return cslist
  else:
    return []

def direction(n):
  rx = n[1]
  ry = -n[0]
  rz = 0.0
  cl = math.sqrt(rx**2 + ry**2)
  if cl == 0:
    rx = 0
    ry = 0
    rz = 1
    phi = 0
    cl = 1
  else:
    rx /= cl
    ry /= cl
    phi = -math.acos(n[2] / math.sqrt(n[0]**2 + n[1]**2 + n[2]**2))
    if n[2] < 0:
      phi = 2.0 * math.pi + phi
  s = "%f %f %f %f" % (rx / cl, ry / cl, rz, phi)
  return s

def tri_str(a,b,c):
  s = "%f %f %f " % (a[0] - b[0], a[1] - b[1], a[2] - b[2])
  s += "%f %f %f " % (b[0], b[1], b[2])
  s += "%f %f %f 0" % (c[0] - b[0], c[1] - b[1], c[2] - b[2])
  return s

def color(a,N):
  basef = math.log(float(N + 2)) / math.log(3.0)
  base = int(basef)
  if basef > base:
    base += 1
  if base < 2:
    base = 2
  a += 1
  v = base * base
  r = int(a / v)
  a -= v * r
  g = int(a / base)
  a -= base * g
  b = a
  ri = int(255.0 * r / (base - 1.0)) * 256
  gi = int(255.0 * g / (base - 1.0)) * 256 * 256
  bi = int(255.0 * b / (base - 1.0)) * 256 * 256 * 256
  return (ri + gi + bi)

if len(sys.argv) < 4:
  print "usage: %s N seed outtype(0:rep, 1:dat)" % (sys.argv[0])
  sys.exit(0)

N = int(sys.argv[1])
np.random.seed(int(sys.argv[2]))
outtype = int(sys.argv[3])

c = np.random.random((N,3))
d = np.random.random(N)
fn = np.zeros((N,3), dtype=float)
for a in range(N):
  ctheta = np.random.random() * 2.0 - 1.0
  stheta = math.sqrt(1.0 - ctheta * ctheta)
  phi = np.random.random() * math.pi * 2.0;
  fn[a][0] = stheta * math.cos(phi)
  fn[a][1] = stheta * math.sin(phi)
  fn[a][2] = ctheta
  d[a] = np.dot(fn[a],c[a])

pa = []
sa = []
lines = []
nodesonplane = []
for a in range(N):
  lines.append([])
  nodesonplane.append([])
  for b in range(N):
    lines[a].append([])
i = 0
for a in range(N):
  for b in range(a + 1,N):
    for c in range(b + 1,N):
      tmpp = np.linalg.solve([fn[a],fn[b],fn[c]],[d[a],d[b],d[c]])
      if tmpp[0] >= 0.0 and tmpp[0] < 1.0 and \
        tmpp[1] >= 0.0 and tmpp[1] < 1.0 and \
	tmpp[2] >= 0.0 and tmpp[2] < 1.0:
	pa.append(tmpp)
	sa.append([a, b, c])
	lines[a][b].append(i)
	lines[b][a].append(i)
	lines[a][c].append(i)
	lines[c][a].append(i)
	lines[b][c].append(i)
	lines[c][b].append(i)
	nodesonplane[a].append(i)
	nodesonplane[b].append(i)
	nodesonplane[c].append(i)
	i += 1
p = np.array(pa,dtype=float)
s = np.array(sa,dtype=float)
del pa
del sa
NP = len(p)
nodes = []
for a in range(NP):
  nodes.append([])

#links
links = []
for a in range(N):
  for b in range(a):
    if len(lines[a][b]) > 1:
      dist = np.zeros(len(lines[a][b]),dtype=float)
      iv = np.cross(fn[a],fn[b])
      for c in range(1,len(lines[a][b])):
        dist[c] = np.dot(iv, p[lines[a][b][c]] - p[lines[a][b][0]])
      sortlist = sorted(range(len(dist)), key=lambda k: dist[k])
      for c in range(len(lines[a][b]) - 1):
        i1 = sortlist[c]
        i2 = sortlist[c + 1]
	nodes[lines[a][b][i1]].append(lines[a][b][i2])
	nodes[lines[a][b][i2]].append(lines[a][b][i1])
	links.append([ lines[a][b][i1], lines[a][b][i2] ])

'''
for a in range(NP):
  print a,nodes[a]
sys.exit(0)
'''
#sides
msides = []
for a in range(N):
  for node in nodesonplane[a]:
    nn = neigh_plane(nodes[node],nodesonplane[a])
    if len(nn) < 2:
      continue
    tmpf = np.dot(p[nn[0]]-p[node],p[nn[1]]-p[node])
    tmpf /= np.linalg.norm(p[nn[0]]-p[node])
    tmpf /= np.linalg.norm(p[nn[1]]-p[node])
    if tmpf < -0.99999999:
      if len(nn) == 2:
	continue
      tmpi = nn[1]
      nn[1] = nn[2]
      nn[2] = tmpi
      cslist = loop(nn[0],node,nn[1],nodes,p,nodesonplane[a])
      if len(cslist) > 2:
	msides.append(cslist)
      cslist = loop(nn[1],node,nn[2],nodes,p,nodesonplane[a])
      if len(cslist) > 2:
	msides.append(cslist)
      if len(nn) == 4:
	cslist = loop(nn[2],node,nn[3],nodes,p,nodesonplane[a])
	if len(cslist) > 2:
	  msides.append(cslist)
	cslist = loop(nn[3],node,nn[0],nodes,p,nodesonplane[a])
	if len(cslist) > 2:
	  msides.append(cslist)

msides.sort()
sides = []
osides = []
for side in msides:
  sside = sorted(side)
  if not sside in osides:
    sides.append(side)
    osides.append(sside)

facesonedges = []
for a in range(NP):
  facesonedges.append([])
  for b in range(NP):
    facesonedges[a].append([])
for si in range(len(sides)):
  side = sides[si]
  for a in range(0,len(side)):
    b = (a + 1) % len(side)
    facesonedges[side[a]][side[b]].append(si)
    facesonedges[side[b]][side[a]].append(si)

'''
for a in range(NP):
  for b in range(NP):
    if len(facesonedges[a][b]):
      print a,b,facesonedges[a][b]
sys.exit(0)
'''

rbodies = []

for si in range(len(sides)):
  side = sides[si]
  nv = np.cross(p[side[1]] - p[side[0]],p[side[2]] - p[side[0]])
  if side[0] < side[1]:
    e = [ side[0], side[1] ]
  else:
    e = [ side[1], side[0] ]
  for f in facesonedges[e[0]][e[1]]:
    if f == si:
      continue
    bodyedgelist = []
    for a in range(1,len(side)):
      b = (a + 1) % len(side)
      if side[a] < side[b]:
	bodyedgelist.append([ 0, side[a], side[b] ])
      else:
	bodyedgelist.append([ 0, side[b], side[a] ])
    nside = sides[f]
    nv2 = np.cross(p[nside[1]] - p[nside[0]],p[nside[2]] - p[nside[0]])
    dotn = np.dot(nv,nv2) / np.linalg.norm(nv) / np.linalg.norm(nv2)
    if abs(dotn) < 0.99999999:
      bodysidelist = [ si, f ]
      for a in range(0,len(nside)):
	b = (a + 1) % len(nside)
	if nside[a] < nside[b]:
	  l = [ nside[a], nside[b] ]
	else:
	  l = [ nside[b], nside[a] ]
        if l != e:
	  bodyedgelist.append([1, l[0], l[1] ])
      ei = 0
      eip = 1
      while len(bodyedgelist):
	found = 0
	for iei in range(len(bodyedgelist)):
	  if bodyedgelist[ei][0] == bodyedgelist[iei][0]:
	    continue
	  for ieip in range(1,3):
	    if bodyedgelist[ei][eip] == bodyedgelist[iei][ieip]:
	      found = 1
	      break
	  if found:
	    break
	if found:
	  ei = 0
	  eip = 1
	  newfaceset = \
	    list(set(facesonedges[bodyedgelist[ei][1]][bodyedgelist[ei][2]])\
	    & set(facesonedges[bodyedgelist[iei][1]][bodyedgelist[iei][2]]))
	  if len(newfaceset) == 0:
	    found = 0
	    break
	  newface = newfaceset[0]
	  bodysidelist.append(newface)
	  nnside = sides[newface]
	  #print "BE",bodyedgelist
	  #add new edges and remove closed edges
	  for a in range(0,len(nnside)):
	    b = (a + 1) % len(nnside)
	    if nnside[a] < nnside[b]:
	      l = [ nnside[a], nnside[b] ]
	    else:
	      l = [ nnside[b], nnside[a] ]
	    add = 1
	    for b in range(len(bodyedgelist)):
	      if bodyedgelist[b][1] == l[0] and bodyedgelist[b][2] == l[1]:
	        bodyedgelist.pop(b)
		add = 0
		break
	    if add:
	      bodyedgelist.append([ len(bodysidelist), l[0], l[1] ])
	  '''
	  print found,ei,eip,iei,ieip,bodyedgelist
	  print "BBB",bodyedgelist[ei], bodyedgelist[iei]
	  print "F1",facesonedges[bodyedgelist[ei][1]][bodyedgelist[ei][2]]
	  print "F2",facesonedges[bodyedgelist[iei][1]][bodyedgelist[iei][2]]
	  print "SIDES",newface,side,nside,nnside,bodysidelist
	  print "BE",bodyedgelist
	  '''
	else:
	  eip += 1
	  if eip == 3:
	    eip = 1
	    ei += 1
	    if ei >= len(bodysidelist):
	      break
    if len(bodyedgelist) == 0:
      rbodies.append(sorted(bodysidelist))

bodies = []
prev = []
for b in sorted(rbodies):
  if b != prev:
    prev = b
    bodies.append(b)
'''
print bodies
sys.exit(0)
print msides
print sides
sys.exit()
'''

#print
if outtype == 0:
  xx = [ [0.0,0.0], [0.0,1.0], [1.0,1.0], [1.0,0.0]]
  print "0 i 12336 i 4294967072 i %d" % (N)
  for a in range(N):
    pdir = np.argmax(np.abs(fn[a]))
    q = np.zeros((4,3),dtype=float)
    for i in range(4):
      xxi = 0
      for b in range(3):
	if b == pdir:
	  q[i][b] += d[a] / fn[a][pdir]
	else:
	  q[i][b] = xx[i][xxi]
	  q[i][pdir] -= xx[i][xxi] * fn[a][b] / fn[a][pdir]
	  xxi += 1
    q *= 2
    q -= [1.0,1.0,1.0]
    print "%f %f %f" % (q[0][0]-q[1][0], q[0][1]-q[1][1], q[0][2]-q[1][2])
    print "%f %f %f" % (q[1][0], q[1][1], q[1][2])
    print "%f %f %f" % (q[2][0]-q[1][0], q[2][1]-q[1][1], q[2][2]-q[1][2])
    print "%f %f %f" % (q[3][0]-q[1][0], q[3][1]-q[1][1], q[3][2]-q[1][2])
  p *= 2
  p -= [1.0,1.0,1.0]
  print "i 34 i 4294967042 i %d" % (len(p))
  for pp in p:
    print 0.1,pp[0],pp[1],pp[2]
  '''
  print "i 99 i 4278190080 i %d" % len(links)
  for l in links:
    pa = p[l[0]]
    pb = p[l[1]]
  #  print "XXX",l[0],l[1],pa,pb,p[l[0]],p[l[1]]
    print 0.02,np.linalg.norm(pa-pb),pa[0],pa[1],pa[2]
    print direction(pb-pa)
  '''
  tri = 0
  for side in sides:
    tri += len(side) - 2
  print "i 111 i 4294967072 i %d" % (tri)
  for side in sides:
    for a in range(1,len(side)-1):
      print tri_str(p[side[0]],p[side[a]],p[side[a+1]])
  i = 0
  for body in bodies:
    numedge = 0
    num = 0
    for side in body:
      num += len(sides[side]) - 2
      numedge += len(sides[side]) 
    print "i 111 i %d i %d" % (color(i, len(bodies)), num)
    for si in body:
      side = sides[si]
      for a in range(1,len(side)-1):
	print tri_str(p[side[0]],p[side[a]],p[side[a+1]])
    print "i 99 i 4278190080 i %d" % (numedge)
    tmpn = 0
    for si in body:
      side = sides[si]
      for a in range(len(side)):
	pa = p[side[a]]
	pb = p[side[(a+1) % len(side)]]
	tmpn += 1
	print 0.02,np.linalg.norm(pa-pb),pa[0],pa[1],pa[2]
	print direction(pb-pa)
    if tmpn != numedge:
      sys.stderr.write("missmatch %d %d\n" % (tmpn,numedge))
    i += 1
  print "i 0"
elif outtype == 1:
  for body in bodies:
    index = range(NP)
    print "object"
    vv = []
    for si in body:
      for pp in sides[si]:
        vv.append(pp)
    vp = list(set(vv))
    for a in range(len(vp)):
      index[vp[a]] = a
    print "vertices %d" % (len(vp))
    for c in vp:
      print p[c][0],p[c][1],p[c][2]
    print "faces %d" % (len(body))
    for si in body:
      side = sides[si]
      ss = ""
      for a in range(len(side)):
        ss += "%d " % (index[side[a]])
      print ss
