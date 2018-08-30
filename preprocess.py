import re, getopt, sys

out = open("out.txt", "w")
patterns = {}
LOGLEVEL = set(["INFO", "FATAL", "ERROR", "WARNING", "SEVERE", "FAILURE"])

def getTrueMT():
	global pattern_psu
	fpMsgT = open("bglMsgTypes", 'r')
	msg_type_id = 1  # line id in fnMsgT, started with 1
	for mt in fpMsgT.readlines():
		mt = mt.replace('(', '\(')  # boot  (command *
		mt = mt.replace(')', '\)')
		mt = mt.replace('[', '\[')  # boot  (command *
		mt = mt.replace(']', '\]')
		mt = mt.replace('.', '\.')
		mt = mt.replace('|', '\|')
		mt = mt.replace('$', '\$')
		if mt.find('(\.*)')<0: mt = mt.replace('*', '(.*)')
		while mt[0]==' ' or mt[0]=='\n' or mt[0]=='\r' or mt[0]=='$': mt=mt[1:]
		while mt[-1]==' ' or mt[-1]=='\n' or mt[-1]=='\r' or mt[-1]=='$': mt=mt[:-1]
	# #	print mt
	# 	while mt.find('...')>=0:
	# 		mt=mt.replace('...', '..')
	# 	mt = mt.replace('..', '...*')
	# 	mt = mt.replace('**', '*')
	# 	for i in range(5):
	# 		mt = mt.replace('* *', '*') # Fan speeds ( * * * * * * )
	# 	mt = mt.replace(' * ', '.*')
	# 	mt = mt.replace(' *', '.*')
	# 	mt = mt.replace('* ', '.*')
	# 	mt = mt.replace('*:*:*:*:*', '.*:.*:.*:.*:.*');
	# 	#		mt = mt.replace('-*', '-.*') #  Targeting domains:node-* and nodes:node-*
	# 	mt = mt.replace('=*', '=.*')
	# 	mt = mt.replace('r*', 'r.*')
	# 	#		mt = mt.replace('\\', '.*') # psu failure\,ambient=27
	# 	mt = mt.replace('(*)', '(.*)')
	# 	mt='.*'+mt+'.*'
	# 	mt = mt.replace('BglCtlPavTrace*', 'BglCtlPavTrace.*')
		print ("processed msg type: ", mt)
		newP = re.compile(mt)
		patterns[newP] = [msg_type_id, []]
		msg_type_id += 1
	empty_msgTypeId = msg_type_id+1
	digit_msgTypeId = empty_msgTypeId+1
	fpMsgT.close()


getTrueMT()
for log in open("bgl2").readlines():
	tmp = log.split(" ")
	
	if tmp[-1] in LOGLEVEL or tmp[-1].find('FATAL')>=0 or tmp[-1].find('INFO')>=0:
		continue
	
	start=0
	while len(tmp)>start and (tmp[start] not in LOGLEVEL):
		start += 1
	
	if len(tmp) > start and tmp[start] in LOGLEVEL:
		log = " ".join(tmp[start+1:])  # not including LOGLEVEL here
	#elif len(tmp) <= start:
#		print "tmp[-1]", tmp[-1], "no log level: ", log
	if log.strip().isdigit():
		continue
	rstP = None
	omStr = ""
	for pp in patterns.keys():
		mobj = pp.match(log)

		if mobj:
			if not rstP:
				rstP = pp
				omStr = pp.pattern
			elif len(pp.pattern) > len(omStr):
				rstP = pp
				omStr = pp.pattern
	
	if rstP:
		#patterns[rstP][1].append(lineId) # 1-indexed
		#logId_msgTypeId[lineId] = patterns[rstP][0]
		out.write(str(patterns[rstP][0]) + "\n")
