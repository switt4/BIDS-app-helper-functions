#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:47:03 2019

@author: Suzanne T. Witt, Ph.D.
"""

import os
import sys
import re
import fnmatch
import json
import pandas as pd
import argparse
import subprocess
import nilearn
import numpy as np
import xml.etree.ElementTree as ET
from glob import glob
from nilearn.image import (load_img, resample_to_img)
import fsl
import fsl.data.featanalysis as FA
import matplotlib.pyplot as plt

def run(command, env={}):
	merged_env = os.environ
	merged_env.update(env)
	process = subprocess.Popen(command, stdout=subprocess.PIPE,
							   stderr=subprocess.STDOUT, shell=True,
							   env=merged_env)
	while True:
		line = process.stdout.readline()
		line = str(line, 'utf-8')[:-1]
		print(line)
		if line == '' and process.poll() != None:
			break
	if process.returncode != 0:
		raise Exception("Non zero return code: %d"%process.returncode)

def read_tsv(inTSV):
	dataTSV = pd.read_table(inTSV)
	dataTSV.head()
	#confoundsHeaders = list(confounds)
	return dataTSV

def read_json(inJSON):
	with open(inJSON,'rt') as cj:
		dataJSON = json.load(cj)
	return dataJSON

def parse_compcor_json(inDict,mask):
	maskCompCor = []
	for key, subdict in inDict.items():
		sublist = list(subdict.values())
		if mask in sublist:
			maskCompCor.append(key)
	return maskCompCor

def confound_plot(timeScale,inConfound,filename,yScale=[],legend=[]):
	fig,ax = plt.subplots()
	ax.plot(timeScale,inConfound)
	ax.legend(legend,loc='upper left')
	ax.set_xlabel('time (sec)')
	ax.set_ylabel(yScale)
	plt.tight_layout()
	plt.savefig(filename,format='svg')
	plt.close()

def confound_scatter(boldSignal,inConfound,filename,numEVs,xScale=[],yScale=[],legend=[]):
	fig,ax = plt.subplots()
	ax.scatter(boldSignal,inConfound)
	ax.set_xlabel(xScale)
	ax.set_ylabel(yScale)
	plt.tight_layout()
	plt.savefig(filename,format='svg')
	plt.close()


def barh_plot(inDict,Title,filename):
	fig,ax = plt.subplots()
	temp = sorted((value,key) for (key,value) in inDict.items())
	sortedDict = dict([(key, value) for (value, key) in temp])
	ax.barh(list(sortedDict.keys()),list(sortedDict.values()),align='center')
	ax.set_title(Title)
	plt.tight_layout()
	plt.savefig(filename,format='svg')
	plt.close()


parser = argparse.ArgumentParser(description='Example BIDS App entrypoint script.')
parser.add_argument('bids_dir', help='The BIDS directory of the input dataset '
					'formatted according to the BIDS standard.')
parser.add_argument('output_dir', help='The directory where the output files '
					'should be stored. If you are running group level analysis '
					'this folder should be prepopulated with the results of the'
					'participant level analysis.')
parser.add_argument('analysis_level', help='Level of the analysis that will be performed. '
					'Multiple participant level analyses can be run independently '
					'(in parallel) using the same output_dir.',
					choices=['participant'])
parser.add_argument('--fmriprep_dir', help='Specify the same output directory as when '
					'fmriprep was run on the dataset.  E.g., there should be a sub-directory '
					'called, "fmriprep" that was created when fmriprep ran.'
					'The script assumes that the fmriprep output directory is '
					'still formatted according to the standard fmriprep output directory.')
parser.add_argument('--feat_dir', help='Specify the directory where the sample design.fsf '
					'file is saved.')
parser.add_argument('--participant_label', help='The label(s) of the participant(s) that should be analyzed. The label '
				   'corresponds to sub-<participant_label> from the BIDS spec '
				   '(so it does not include "sub-"). If this parameter is not '
				   'provided all subjects should be analyzed. Multiple '
				   'participants can be specified with a space separated list.',
				   nargs = "+")
parser.add_argument('--task_label', help='Enter the "task-<task_label>" label for the '
					'task you wish to analyze.  All runs of this task '
					'must have a valid "events.tsv" files.',
					nargs = "+")
parser.add_argument('--HarvardOxford_region', help='Specify region from Harvard Oxford cortical atlas to test. '
					'Use the value in $FSLDIR/data/atlases/HarvardOxford-Cortical.xml; '
					'correction for values starting at 0 are applied automatically. '
					'If parameter is not set, the "Intracalcarine Cortex" (label = 23) will '
					'be tested as default. Region name should be specified in single quotes.',
					nargs="+")
#parser.add_argument('-v', '--version', action='version',
#					version='BIDS-App example version {}'.format(__version__))


args = parser.parse_args()

if args.fmriprep_dir is None:
	sys.exit('Error: You must specify an fmnriprep directory.')

if args.feat_dir is None:
	sys.exit('Error: You must specify a directory containing a sample design.fsf file.')

if args.task_label is None:
	sys.exit('Error: You must specify a single task to analyze.')
else:
	taskLabel = str(args.task_label[0])

if not os.path.exists(os.path.join(args.feat_dir,"task-%s"%taskLabel)):
	os.makedirs(os.path.join(args.feat_dir,"task-%s"%taskLabel))

# load in Harvard Oxford atlases and prepare for region extraction
atlasCortexLabels = []
atlasCortexFile = os.path.join('$FSLDIR','data','atlases','HarvardOxford-Cortical.xml')
atlasCortex = ET.parse(os.path.expandvars(atlasCortexFile))
atlasCortexRoot = atlasCortex.getroot()
for roi in range(len(atlasCortexRoot[1])):
	atlasCortexLabels.append(atlasCortexRoot[1][roi].text)

# get full path to $FSLDIR
FSLDIR = os.path.expandvars('$FSLDIR')

# get selected region from Harvard Oxford atlas
if args.HarvardOxford_region:
	testRegionNumber = atlasCortexLabels.index(args.HarvardOxford_region[0]) + 1
else:
	testRegionNumber = atlasCortexLabels.index('Intracalcarine Cortex') + 1

roiRegionNumber = testRegionNumber - 1

subjectsToAnalyze = []
# only for a subset of subjects
if args.participant_label:
	subjectsToAnalyze = args.participant_label
# for all subjects
else:
	subject_dirs = glob(os.path.join(args.bids_dir, "sub-*"))
	subjectsToAnalyze = [subject_dir.split("-")[-1] for subject_dir in subject_dirs]

if not os.path.exists(os.path.join(args.output_dir,"temp")):
	os.makedirs(os.path.join(args.output_dir,"temp"))


# running participant level
if args.analysis_level == "participant":

	# find all func files and calculate effect of denoising them
	for subjectLabel in subjectsToAnalyze:
		print('starting subject:%s'%subjectLabel)
		for funcFile in glob(os.path.join(args.fmriprep_dir,"fmriprep","sub-%s"%subjectLabel,"func","*_bold.nii*")) + glob(os.path.join(args.fmriprep_dir,"fmriprep","sub-%s"%subjectLabel,"ses-*","func","*_bold.nii*")):
			print('working on functional run: %s'%os.path.split(funcFile)[-1])
			
			
	





	