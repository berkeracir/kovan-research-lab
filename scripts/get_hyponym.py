""" 
Script for printing/getting subclass names of a given class from ImageNet Dataset
	
Input: [wnid] of class

Output: text file containing class names
"""

import sys
import os
import subprocess

if len(sys.argv) < 2:
	print "Wrong usage: python", sys.argv[0], "[wnid]"
	exit(1)

HYPONYM_URL = "http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid="
WORDS_URL = "http://www.image-net.org/api/text/wordnet.synset.getwords?wnid="
output_file = open(sys.argv[1] + "_hyponym.txt", 'w')

def return_words(wnid):
	url = WORDS_URL + wnid
	p = subprocess.Popen("curl -s -L " + url, stdout=subprocess.PIPE, shell=True)
	(output, err) = p.communicate()
	p_status = p.wait()
	
	return output.replace('\n',', ').replace('\r','')[:-2]

def rec_get_wnid(wnid, depth):
	url = HYPONYM_URL + wnid
	p = subprocess.Popen("curl -s -L " + url, stdout=subprocess.PIPE, shell=True)
	(output, err) = p.communicate()
	p_status = p.wait()
	
	output_list = output.replace('\n','').split('\r')[:-1]
	prefix = '-' * depth
	output_list = map(lambda x: prefix+x, output_list)
	
	if len(output_list) == 1:
		print '-' * output_list[0].count('-') + return_words(output_list[0].replace('-','')) + " (" + output_list[0].replace('-','') + ")"
		output_file.write('\t' * output_list[0].count('-') + " " + return_words(output_list[0].replace('-','')) + " (" + output_list[0].replace('-','') + ")\n")
	else:
		print '-' * output_list[0].count('-') + return_words(output_list[0].replace('-','')) + " (" + output_list[0].replace('-','') + ")"
		output_file.write('\t' * output_list[0].count('-') + " " + return_words(output_list[0].replace('-','')) + " (" + output_list[0].replace('-','') + ")\n")
		for i in range(1, len(output_list)):
			rec_get_wnid(output_list[i].replace('-',''), depth+1)

rec_get_wnid(sys.argv[1], 0)
output_file.close()
