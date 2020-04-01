#!/usr/bin/python

import math
import sys
from subprocess import Popen, PIPE, STDOUT
import os
import numpy as np

def over_tri(p,a,b,c):
  n0 = np.cross(b-a,c-a)
  n1 = np.cross(b-a,p-a)
  n2 = np.cross(c-b,p-b)
  n3 = np.cross(a-c,p-c)
  return np.dot(n1,n0) >= 0 and np.dot(n2,n0) >= 0 and np.dot(n3,n0) >= 0

def over_vert(p,a,c1):
  return np.dot(a - p,a - c1) > 0

def calc_V(b,c,d):
  return np.abs(np.dot(np.cross(b,c),d)) / 6.0

def calc_TKP(c,faces):
  TKP = np.zeros(3,dtype=float)
  O = np.mean(c, axis = 0)
  p = c - O
  V = 0.0
  for face in faces:
    for a in range(1,len(face)-1):
      tmpV = calc_V(p[face[0]],p[face[a]],p[face[a+1]])
      V += tmpV
      TKP += (p[face[0]] + p[face[a]] + p[face[a+1]]) * tmpV / 4.0
  TKP /= V
  TKP += O
  return [ TKP, V ]

def SU(TKP, c, faces):
  S = 0
  edges = []
  for a in range(len(c)):
    edges.append([])
  for face in faces:
    onface = 0
    for a in range(1,len(face)-1):
      onface += over_tri(TKP,c[face[0]],c[face[a]],c[face[a+1]])
    S += (onface > 0)
    for a in range(len(face)):
      edges[face[a]].append( face[(a+1)%len(face)] )
      edges[face[(a+1)%len(face)]].append( face[a] )

  for a in range(len(c)):
    edges[a] = list(set(edges[a]))
  U = 0
  for i in range(len(c)):
    oververt = 1
    for edge in edges[i]:
      oververt *= over_vert(TKP,c[i],c[edge])
    U += oververt
  return [ S, U ]

def print_SUAB(c,faces):
  [ TKP, V ] = calc_TKP(c,faces)
  maxc = np.zeros(len(c),dtype=float)
  for i in range(len(c)):
    maxc[i] = max(np.linalg.norm(c - c[i], axis=1))
  c1 = np.argmax(maxc)
  c2 = np.argmax(np.linalg.norm(c - c[c1], axis=1))
  C = abs(np.linalg.norm(c[c2] - c[c1]))
  nc = c[c2] - c[c1]
  nc /= np.linalg.norm(nc)
  maxb = np.zeros(len(c),dtype=float)
  for i in range(len(c)):
    nb = np.cross(np.cross(nc, c[i] - c[c1]),nc)
    if np.linalg.norm(nb) == 0 or np.linalg.norm(np.cross(nb,nc)) == 0:
      continue
    nb /= np.linalg.norm(nb)
    nbd = np.dot(c - c[c1], nb)
    maxb[i] = np.max(nbd) - np.min(nbd)
  b1 = np.argmax(maxb)
  nb = np.cross(np.cross(nc, c[b1] - c[c1]),nc)
  nb /= np.linalg.norm(nb)
  B = np.max(maxb)
  na = np.cross(nb,nc)
  na /= np.linalg.norm(na)
  nad = np.dot(c - c[c1], na)
  A = np.max(nad)-np.min(nad)

  [ S, U ] = SU(TKP, c, faces)
  hs = np.zeros(20,dtype = int)
  hu = np.zeros(30,dtype = int)
  for a in range(100):
    ra = np.random.normal(0.0, A * sigma) * na
    rb = np.random.normal(0.0, B * sigma) * nb
    rc = np.random.normal(0.0, C * sigma) * nc
    tmpTKP = TKP + ra + rb + rc
    [ tS, tU ] = SU(tmpTKP, c, faces)
    hs[tS] += 1
    hu[tU] += 1
  sface = np.zeros(len(faces),dtype=float)
  i = 0
  for face in faces:
    for j in range(1,len(face)-1):
      sface[i] += np.linalg.norm(np.cross(c[face[j]] - c[face[0]],\
        c[face[j + 1]] - c[face[0]]))
    i += 1
  Alim = A * B * facelim
  csface = sface.tolist()
  cfaces = list(faces)
  numc = len(c)
  numf = len(faces)
  while min(csface) < Alim and len(csface) > 4:
    mf = np.argmin(csface)
    numc -= len(cfaces[mf]) - 1
    numf -= 1
    newc = cfaces[mf][0]
    for fi in range(len(cfaces)):
      if fi == mf:
        continue
      ci = 0
      was = 0
      while ci < len(cfaces[fi]):
        if cfaces[fi][ci] in cfaces[mf]:
	  if was == 0:
	    cfaces[fi][ci] = newc
	    was = 1
	    ci += 1
	  else:
	    del cfaces[fi][ci]
	else:
	  ci += 1
    del cfaces[mf]
    del csface[mf]
    fi = 0
    while fi < len(cfaces):
      if len(cfaces) < 3:
	numf -= 1
        del cfaces[fi]
        del csface[fi]
      else:
        fi += 1
  cc = []
  nface = np.zeros(10,dtype=int)
  for face in cfaces:
    if len(face) < 10:
      nface[len(face)] += 1
    for csucs in face:
      cc.append(csucs)

  if len(set(cc)) > 3 and len(cfaces) > 3:
    print S,U,A/C,B/C,C,len(set(cc)),len(cfaces),nface[3],nface[4],nface[5],\
      nface[6],nface[7],nface[8],nface[9],V,"X",hs[0],hs[1],hs[2],hs[3],\
      hs[4],hs[5],hs[6],hs[7],hs[8],"Y",hu[0],hu[1],hu[2],hu[3],hu[4],\
      hu[5],hu[6],hu[7],hu[8],hu[9],hu[10],hu[11],hu[12]

def break_it(startc,startfaces,pb,sizelim):
  clist = [ startc ]
  faceslist = [ startfaces ]
  while len(clist):
    c = clist.pop(0)
    faces = faceslist.pop(0)
    maxc = np.zeros(len(c),dtype=float)
    for i in range(len(c)):
      maxc[i] = max(np.linalg.norm(c - c[i], axis=1))
    c1 = np.argmax(maxc)
    c2 = np.argmax(np.linalg.norm(c - c[c1], axis=1))
    C = abs(np.linalg.norm(c[c2] - c[c1]))
    nc = c[c2] - c[c1]
    nc /= np.linalg.norm(nc)
    maxb = np.linalg.norm(np.cross(nc, c - c[c1]),axis=1)
    b1 = np.argmax(maxb)
    na = np.cross(nc, c[b1] - c[c1])
    na /= np.linalg.norm(na)
    nb = np.cross(nc,na)
    nb /= np.linalg.norm(nb)
    nbd = np.dot(c - c[c1],nb)
    B = np.max(nbd)-np.min(nbd)
    nad = np.dot(c - c[c1],na)
    A = np.max(nad)-np.min(nad)
    p_break = 1.0 - A * B / C / C
    if A / C * B / C < np.random.random() * pb and A * B * C > sizelim:
      C_break = np.random.normal(C / 2, C / 5)
      while C_break < 0 or C_break > C:
	C_break = np.random.normal(C / 2, C / 5)
      c_break = c[c1] + nc * C_break
      n_break = nc * C + nb * B + na * A
      n_break *= np.random.random(3)
      n_break /= np.linalg.norm(n_break)
      tedges = []
      edges = []
      lc = len(c)
      for face in faces:
	for a in range(len(face)):
	  ea = min(face[a], face[(a+1)%len(face)])
	  eb = max(face[a], face[(a+1)%len(face)])
	  tedges.append( ea * lc + eb )
      for e in list(set(tedges)):
	edges.append([ int(e/lc), e%lc ])
      newc = []
      newe = []
      for e in edges:
	t = c[e[1]] - c[e[0]]
	t /= np.linalg.norm(t)
	sd = np.dot(n_break, c_break)
	l = (sd - np.dot(n_break, c[e[0]])) / np.dot(n_break, t)
	if l > 0 and l < np.linalg.norm(c[e[1]] - c[e[0]]):
	  newc.append(c[e[0]] + l * t)
	  newe.append(e)
      newc = np.array(newc)
      cl = []
      cr = []
      pp = np.dot(newc[0],n_break)
      for a in range(len(c)):
	if np.dot(c[a],n_break) > pp:
	  cr.append(a)
	else:
	  cl.append(a)
      facel = []
      facer = []
      newedgel = []
      newedger = []
      for face in faces:
	inl = 0
	for f in face:
	  if f in cl:
	    inl |= 1
	  else:
	    inl |= 2
	if inl == 1:
	  tmp = []
	  for f in face:
	    tmp.append(cl.index(f))
	  facel.append(tmp)
	elif inl == 2:
	  tmp = []
	  for f in face:
	    tmp.append(cr.index(f))
	  facer.append(tmp)
	else:
	  tmpl = []
	  tmpr = []
	  for a in range(len(face)):
	    b = (a + 1) % len(face)
	    ainl = face[a] in cl
	    binl = face[b] in cl
	    if ainl:
	      tmpl.append(cl.index(face[a]))
	    else:
	      tmpr.append(cr.index(face[a]))
	    if ainl ^ binl:
	      e = [min(face[a], face[b]), max(face[a], face[b])]
	      inewe = newe.index(e)
	      tmpl.append(len(cl) + inewe)
	      tmpr.append(len(cr) + inewe)
	  for a in range(len(tmpl)):
	    b = (a + 1) % len(tmpl)
	    if tmpl[a] >= len(cl) and tmpl[b] >= len(cl):
	      newedgel.append([min(tmpl[a], tmpl[b]), max(tmpl[a], tmpl[b]) ])
	  for a in range(len(tmpr)):
	    b = (a + 1) % len(tmpr)
	    if tmpr[a] >= len(cr) and tmpr[b] >= len(cr):
	      newedger.append([min(tmpr[a], tmpr[b]), max(tmpr[a], tmpr[b]) ])
	  facel.append(tmpl)
	  facer.append(tmpr)
      newfaceedges = []
      for a in range(len(newc) + len(c)):
	for b in range(a):
	  if [b, a] in newedgel:
	    newfaceedges.append([b,a])
      newface = []
      i = len(cl)
      while len(newfaceedges):
	for a in range(len(newfaceedges)):
	  if newfaceedges[a][0] == i:
	    j = newfaceedges[a][1]
	    newfaceedges.pop(a)
	    break
	  if newfaceedges[a][1] == i:
	    j = newfaceedges[a][0]
	    newfaceedges.pop(a)
	    break
	newface.append(j)
	i = j
      facel.append(newface)
  #    newface = range(len(cr), len(cr) + len(newc))
  #    facer.append(newface)
      newfaceedges = []
      for a in range(len(newc) + len(c)):
	for b in range(a):
	  if [b, a] in newedger:
	    newfaceedges.append([b,a])
      newface = []
      i = len(cr)
      while len(newfaceedges):
	for a in range(len(newfaceedges)):
	  if newfaceedges[a][0] == i:
	    j = newfaceedges[a][1]
	    newfaceedges.pop(a)
	    break
	  if newfaceedges[a][1] == i:
	    j = newfaceedges[a][0]
	    newfaceedges.pop(a)
	    break
	newface.append(j)
	i = j
      facer.append(newface)
      #idaig
      clc = []
      crc = []
      for ci in cl:
	clc.append(c[ci])
      for ci in cr:
	crc.append(c[ci])
      for cc in newc:
	clc.append(cc)
	crc.append(cc)
      clc = np.array(clc)
      crc = np.array(crc)
      clist.append(clc)
      faceslist.append(facel)
      clist.append(crc)
      faceslist.append(facer)

    else:
      # too small or did not break
      print_SUAB(c,faces)

if len(sys.argv) < 8:
  print "usage: %s fin old(0)new(1) seed pb sizelim sigma facelim" % (sys.argv[0])
  sys.exit(0)

oldnew = int(sys.argv[2])
np.random.seed(int(sys.argv[3]))
TKP = np.zeros(3,dtype=float)
pb = float(sys.argv[4])
sizelim = float(sys.argv[5])
sigma = float(sys.argv[6])
facelim = float(sys.argv[7])
oi = -1
state = 0
s = ""
f = open(sys.argv[1],"r")
for line in f:
  if line[0] == '#':
    continue
  m = line.split()
  if state == 1 and len(m) > 2:
    # vertices
    c[i][0] = float(m[0])
    c[i][1] = float(m[1])
    c[i][2] = float(m[2])
    i += 1
  if state == 2 and len(m) > 2:
    # faces
    faces.append([])
    if oldnew:
      for a in range(len(m)):
	faces[i].append( int(m[a]) )
    else:
      for a in range(3,len(m)):
	faces[i].append( int(m[a]) )
    i += 1
  if state == 3 and len(m) > 2:
    # edges
    i += 1
  if line == "object\n":
    if (oi>=0):
      break_it(c,faces,pb,sizelim)
    oi += 1
    faces = []
    c = []
    state = 0;
  elif m[0] == "vertices":
    c = np.zeros((int(m[1]),3),dtype=float)
    state = 1
    i = 0
  elif m[0] == "faces":
    state = 2
    i = 0
  elif m[0] == "edges":
    state = 3
    i = 0
f.close()
break_it(c,faces,pb,sizelim)
