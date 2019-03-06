import models
import tensorflow as tf
import argparse
import os, sys
import config
import utils
import numpy as np
import mir_eval

def train(_):
	model = models.DeepConvSep()
	model.train()

def eval(file_name, plot):
	model = models.DeepConvSep()
	model.test_file(file_name, plot)


	# utils.save_scores_mir_eval(scores, save_path)
	# import pdb;pdb.set_trace()


if __name__ == '__main__':
	if len(sys.argv)<2 or sys.argv[1] == '-help' or sys.argv[1] == '--help' or sys.argv[1] == '--h' or sys.argv[1] == '-h':
		print("%s --help or -h or --h or -help to see this menu" % sys.argv[0])
		print("%s --train or -t or --t or -train to train the model" % sys.argv[0])
		print("%s -e or --e or -eval or --eval  <filename> to evaluate an hdf5 file, add -p for plots" % sys.argv[0])

	else:
		if sys.argv[1] == '-train' or sys.argv[1] == '--train' or sys.argv[1] == '--t' or sys.argv[1] == '-t':
			print("Training")
			tf.app.run(main=train)
		elif sys.argv[1] == '-e' or sys.argv[1] == '--e' or sys.argv[1] == '--eval' or sys.argv[1] == '-eval':

			if "-p" in sys.argv:
				plot = True 
			else:
				plot = False

			if len(sys.argv)<3:
				print("Please give a file to evaluate")
			else:
				file_name = sys.argv[2]
				if not file_name.endswith('.hdf5'):
					file_name = file_name+'.hdf5'
				if not file_name in os.listdir(config.feats_dir):
					print("Currently only supporting hdf5 files which are in the dataset, will be expanded later.")
				else:
					eval(file_name, plot)



		elif sys.argv[1] == '-he' or sys.argv[1] == '--he':
			print("Evaluating, please wait.")
			if len(sys.argv)<5:
				print("Please give path for orignal csvs, estimated csvs and a filename to save the results")
			else:
				path_ori= sys.argv[2]
				path_est = sys.argv[3]
				save_path = sys.argv[4]
				eval_helena(path_ori, path_est, save_path)

