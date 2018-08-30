import re, getopt, sys

fnMsgT = "bglMsgTypes"
fnLogs = "BGL_MERGED.log"
outFile = "BGL_lineId2MsgTypeId_withLabel.txt"
nTotalLogs = 0

# fnMsgT = 'floattemp'
#fnLogs = "BGL_normal.log"
#outFile = "BGL_normal_lineId2MsgTypeId.txt"


#fnMsgT = "test_bglMsgTypes"
#fnLogs = "test_log"
trueRst = "true_rsts"
fnLcsRst = "bgl2_lcs_rsts"
fnIplomRst = "tmpi"
#fnLpRsts = [fnLcsRst, fnIplomRst]
LOGLEVEL = set(["INFO", "FATAL", "ERROR", "WARNING", "SEVERE", "FAILURE"])

final_rsts = {} # msg type: list of line ids
#patterns = {} # compiled re : [log ids having this message type]
patterns = {} # compiled re : [message_type_id_in_fileREs, [log ids having this message type]]
logId_msgTypeId = {} # line id in original log file : message_type_id_in_fileREs   -- to map each log line into a log key
logId_label = {}
empty_msgTypeId, digit_msgTypeId = 0, 0

pattern_psu = None
empty_logs=[] # store log ids having empty lines
digit_logs=[]

# logic: for each log, match it with re in patterns[]; put the log id to the longest re.

def diff(a, b):
	b = set(b)
	return [aa for aa in a if aa not in b]

# read in message types and compile to re
def getTrueMT():
	global pattern_psu
	fpMsgT = open(fnMsgT, 'r')
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
 
# read in each log and match with patterns[]
def matchLogMT():
	global pattern_psu, nTotalLogs
	fpLogs = open(fnLogs, 'r')
	allLogs = fpLogs.readlines()
	nTotalLogs = len(allLogs)
	lineId = 0
	for log in allLogs:
#		print lineId
#		while len(log)>0 and (log[0]==' ' or log[0]=='\n'): log=log[1:]
#		while len(log)>0 and (log[-1]==' ' or log[-1]=='\n'): log=log[:-1]
#		if len(log)==0:
#			lineId += 1
#			empty_logs.append(lineId) # assign a log id for empty_logs?
#			continue
		lineId += 1
		tmp = log.split(" ")
		if tmp[0] == '-':
			logId_label[lineId] = 1
		else:
			logId_label[lineId] = -1
		if tmp[-1] in LOGLEVEL or tmp[-1].find('FATAL')>=0 or tmp[-1].find('INFO')>=0:
			empty_logs.append(lineId)
			logId_msgTypeId[lineId] = empty_msgTypeId
			continue
		start=0
		while len(tmp)>start and (tmp[start] not in LOGLEVEL):
			start += 1
		if len(tmp) > start and tmp[start] in LOGLEVEL:
			log = " ".join(tmp[start+1:])  # not including LOGLEVEL here
		elif len(tmp) <= start:
			print "tmp[-1]", tmp[-1], "no log level: ", log
#		if len(log)==0:
#			empty_logs.append(lineId)
#			logId_msgTypeId[lineId] = empty_msgTypeId
#			continue
		if log.strip().isdigit():
			digit_logs.append(lineId) # assign a log id for digit_logs?
			logId_msgTypeId[lineId] = digit_msgTypeId
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
		if not rstP:
			print "not matched log: ", log
		else:
#			print "lineId, rstP", lineId, rstP.pattern
			patterns[rstP][1].append(lineId) # 1-indexed
			logId_msgTypeId[lineId] = patterns[rstP][0]
	print "total matched log: ", len(logId_msgTypeId), "logId_msgTypeId", logId_msgTypeId

def lineIdToFile():
	global nTotalLogs
	print 'nTotalLogs ', nTotalLogs
	with open(outFile, 'w') as fp:
		for i in range(nTotalLogs):
			if i+1 in logId_msgTypeId:
				fp.write(str(logId_msgTypeId[i+1])+' '+str(logId_label[i+1])+'\n')
			else:
				fp.write('line '+str(i+1)+' not found\n')

# fun: to translate original log file to a set of log sequences (for LSTM input)
def genLogSeqs():
	fnSeqs = fnLogs+'Seqs'
	fpSeqs = open(fnSeqs, 'w')
	lineId = 0
	with open(fnLogs, 'r') as fpLogs:
		for log in fpLogs.readlines():
			fpSeqs.write(str(logId_msgTypeId[lineId])+" ")
			fpSeqs.write('\n')
			lineId += 1
	fpSeqs.close()

# print results
def sumTrueRsts():
	print "Now print the matching results: "
	for pp, lids in patterns.items():
		final_rsts[pp.pattern] = lids
	#	print "message type: ", pp.pattern
	#	print "size: ", len(lids), "line ids: ", lids
	#	print "size: ", len(lids), "line ids: ", lids
	#print "message type: psu failure\\"
	#print "size: ", len(pattern_psu), pattern_psu

	for mt, lids in final_rsts.items():
	#	final_rsts[mt] = map(lambda x: x-1, lids) # to 0-indexed in order to compare with lcs/iplom results
		lids=sorted(lids)
		print "message:", mt
		print "size:", len(lids)
		print "lineIds:", lids
		print " "

def getTrueRsts():
	with open(trueRst) as fpRst:
		message=None
		size=0 # for debug
		for line in fpRst.readlines():
			if line.find("message: ")==0:
				message = line.split("message: ")[1]
			elif line.find("size:")==0:
				size = int(line.split(" ")[1])
			elif line.find("lineIds:")==0:
				lineIds = line.split(" ")[1:]
				if lineIds[-1][-1]=='\n' or lineIds[-1][-1]==' ':
					lineIds[-1]=lineIds[-1][:-1]
				if len(lineIds[-1])==0:
					lineIds = lineIds[:-1]
#				for i in range(len(lineIds)):
#					if lineIds[i][-1]==',':
#						lineIds[i] = lineIds[i][:-1]
				lineIds = map(lambda x:int(x[:-1] if x[-1]==',' else x), lineIds)
				if(size!=len(lineIds)):
					print "true, line id count not equal to size, size: ", size, "len(lineIds): ", len(lineIds), "lineIds: ", lineIds
#					assert(0)
				lineIds=sorted(lineIds)
				final_rsts[message] = lineIds


# parse lcs result
lcs_rsts = {} # message type : line ids
def getLpRsts(fnLpRst):
	with open(fnLpRst) as fpLCS:
		message=None
		size=0 # for debug
		for line in fpLCS.readlines():
			if line.find("message: ")==0:
#				print "lcs message1:", line
				message = line.split("message: ")[1][:-1]
#				print "lcs message2:", message
			elif line.find("size:")==0:
				size = int(line.split(" ")[1])
			elif line.find("lineIds:")==0:
				lineIds = line.split(" ")[1:-1]
				lineIds = map(lambda x:int(x), lineIds)
				if(size!=len(lineIds)):
					print "lcs, line id count not equal to size, size: ", size, "len(lineIds): ", len(lineIds), "lineIds: ", lineIds, "message: ", message
					assert(0)
				lineIds=sorted(lineIds)
				lcs_rsts[message] = lineIds
	print "lcs_rsts length:", len(lcs_rsts)

# compare lcs_rsts with ground truth final_rsts:
def getAcc():
	totalCnt, accCnt = 0, 0
	for mt, lids in final_rsts.items():
#		if mt.find("instruction")>=0:
#			print "true msg type:", mt
#			print "true lids size:", len(lids)
		matched = False
		totalCnt += len(lids)
		for lm, ll in lcs_rsts.items():
#			if lm.find("instruction")>=0:
#				print "lcs msg type:", lm
#				print "lcs lids size:", len(ll)
#				print "true-lcs:", diff(lids, ll), "lcs-true:", diff(ll, lids)
			if len(lids)==len(ll) and lids==ll:
				matched=True
				print "equal, true message type: ", mt, " lcs message type: ", lm
				accCnt += len(ll)
				break
		if not matched:
			print "true message type not found: ", mt, " not found size: ", len(lids)

	accuracy = float(accCnt) / float(totalCnt)
	print "lcs accuracy: ", accuracy

def printHelp():
	print "python matchLogToMsgTypes.py -i <logParserRstFile> -o <accuracyOutputFile>"

def main(argv):
	global fnLogs
	parsedFile = ''
	outputfile = ''
	try:
		opts, args = getopt.getopt(argv, "hp:l:", ["parser=", "log="])
	except getopt.GetoptError:
		pass
#		printHelp()
#		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			printHelp()
			sys.exit()
		elif opt in ("-p", "--parser"):
			parsedFile = arg
		elif opt in ("-l", "--log"):
			fnLogs = arg
#	print 'parser parsed result file is: ', parsedFile
	print 'original log file is: ', fnLogs
	getTrueMT()
	matchLogMT()

	lineIdToFile()

#	genLogSeqs()

#	sumTrueRsts()
#	getLpRsts(parsedFile)
#	getAcc()

if __name__ == "__main__":
	main(sys.argv[1:])
