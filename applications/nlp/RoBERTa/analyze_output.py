import numpy as np
import pandas as pd
import sys
import glob
import torch
import torch.nn

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger

#from moses.script_utils import  read_smiles_csv

seq_length = 57
vocab_length = 30
  
def detokenize(inp,vocab):
  output = ""
  for i in inp:
    token = list(vocab.keys())[list(vocab.values()).index(int(i))]
    if(token[0]!='<'):
      output = output+token

  return output

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def read_input_from_callback(fdir):

	input_files = glob.glob(fdir+"*_output0.csv")
	df = pd.concat((pd.read_csv(f,header=None) for f in input_files))
	#print(df.shape)

	df.to_csv("inps.csv", index=False, header=False)

def read_pred_from_callback(fdir):

	input_files = glob.glob(fdir+"*_output0.csv")

	ins = np.loadtxt(input_files[0], delimiter=",")
	for i, f in enumerate(input_files):
		if(i > 0) :
			ins = np.concatenate((ins, np.loadtxt(f,delimiter=",")))

	num_cols = ins.shape[1]
	#print("Num cols ", num_cols)
	num_samples = ins.shape[0]
	#print("Num samples ", num_samples)

	df = pd.DataFrame(columns=range(0,seq_length-1))
	#print(df.shape)


	for j in range(num_samples):
		preds_sm = []
		for i in range(seq_length-1):
				preds = softmax(ins[j,vocab_length*i:vocab_length*(i+1)])

				preds = torch.from_numpy(preds)
				preds = torch.argmax(preds).item()
				preds_sm.append(preds)
				
		#print([preds_sm])
		df = df.append(pd.DataFrame([preds_sm]))

	#print(df.shape)
	df.to_csv("preds.csv", index=False, header=False)


def get_smiles_from_lbann_tensors(fdir, vocab_path):


  ###################
  # First input files 
  ###################
 
  input_files = glob.glob(fdir+"inps.csv")

  ins = np.loadtxt(input_files[0], delimiter=",")
  for i, f in enumerate(input_files):
    if(i > 0) :
      ins = np.concatenate((ins, np.loadtxt(f,delimiter=",")))

  num_cols = ins.shape[1]
  #print("Num cols ", num_cols)
  num_samples = ins.shape[0]
  #print("Num samples ", num_samples)

  vocab = pd.read_csv(vocab_file, delimiter=" ", header=None).to_dict()[0]
  vocab = dict([(v,k) for k,v in vocab.items()])
  #print(vocab)


  samples = [detokenize(i_x,vocab) for i_x in ins[:,0:]] 

  #print(samples[0]) 

  samples = pd.DataFrame(samples, columns=['SMILES'])
  

  #print("Save gt files to " , "gt_"+"smiles.txt")
  samples.to_csv("gt_"+"smiles.txt", index=False)

  ####################
  # Second input files 
  ####################

  input_files = glob.glob(fdir+"preds.csv")

  ins = np.loadtxt(input_files[0], delimiter=",")
  for i, f in enumerate(input_files):
    if(i > 0) :
      ins = np.concatenate((ins, np.loadtxt(f,delimiter=",")))

  num_cols = ins.shape[1]
  #print("Num cols ", num_cols)
  num_samples = ins.shape[0]
  #print("Num samples ", num_samples)

  vocab = pd.read_csv(vocab_file, delimiter=" ", header=None).to_dict()[0]
  vocab = dict([(v,k) for k,v in vocab.items()])
  #print(vocab)


  samples = [detokenize(i_x,vocab) for i_x in ins[:,0:]] 

  #print(samples[0]) 

  samples = pd.DataFrame(samples, columns=['SMILES'])
  
  #print("Save gt files to " , "pred_"+"smiles.txt")
  samples.to_csv("pred_"+"smiles.txt", index=False)
          
def compare_decoded_to_original_smiles(orig_smiles, decoded_smiles, output_file=None):
    """
    Compare decoded to original SMILES strings and output a table of Tanimoto distances, along with
    binary flags for whether the strings are the same and whether the decoded string is valid SMILES.
    orig_smiles and decoded_smiles are lists or arrays of strings.
    If an output file name is provided, the table will be written to it as a CSV file.
    Returns the table as a DataFrame.

    """
    res_df = pd.DataFrame(dict(original=orig_smiles, decoded=decoded_smiles))
    is_valid = []
    is_same = []
    tani_dist = []
    accuracy = []
    count = 0
    data_size = len(orig_smiles)
    for row in res_df.itertuples():
        count = count + 1
        #compute char by char accuracy
        hit = 0
        for x, y in zip(row.original, row.decoded):
            if x == y:
                hit = hit+1
        accuracy.append((hit/len(row.original))*100)

        is_same.append(int(row.decoded == row.original))
        orig_mol = Chem.MolFromSmiles(row.original)
        if orig_mol is None:
          #Note, input may be invalid, if original SMILE string is truncated 
          is_valid.append('x')
          tani_dist.append(-1)
          continue
        dec_mol = Chem.MolFromSmiles(row.decoded)
        RDLogger.DisableLog('rdApp.*')
        if dec_mol is None:
            is_valid.append(0)
            tani_dist.append(1)
        else:
            is_valid.append(1)
            orig_fp = AllChem.GetMorganFingerprintAsBitVect(orig_mol, 2, 1024)
            dec_fp = AllChem.GetMorganFingerprintAsBitVect(dec_mol, 2, 1024)
            tani_sim = DataStructs.FingerprintSimilarity(orig_fp, dec_fp, metric=DataStructs.TanimotoSimilarity)
            tani_dist.append(1.0 - tani_sim)
    res_df['is_valid'] = is_valid
    res_df['is_same'] = is_same
    res_df['smile_accuracy'] = accuracy
    res_df['tanimoto_distance'] = tani_dist
    global_acc  = np.mean(np.array(accuracy))
    res_df['total_avg_accuracy'] = [global_acc]*len(accuracy)
    
    print("Mean global accuracy % ", global_acc)
    print("Validity % ", (is_valid.count(1)/data_size)*100)
    print("Same % ", (is_same.count(1)/data_size)*100)
    valid_tani_dist = [ t for t in tani_dist if t >= 0 ] 
    print("Average tanimoto ", np.mean(np.array(valid_tani_dist)))
    

    if output_file is not None:
        output_columns = ['original', 'decoded', 'is_valid', 'is_same', 'smile_accuracy','tanimoto_distance','total_avg_accuracy']
        res_df.to_csv(output_file, index=False, columns=output_columns)
    return(res_df)

def read_smiles_csv(path):
    return pd.read_csv(path,
                       usecols=['SMILES'],
                       squeeze=True).astype(str).tolist()


fdir_ouput='/g/g92/tran71/tran71/new_script/output/'
fdir_input='/g/g92/tran71/tran71/new_script/input/'

vocab_file= 'vocab_train.txt'
fdir='/g/g92/tran71/tran71/new_script/'

read_input_from_callback(fdir_input)
read_pred_from_callback(fdir_ouput)

get_smiles_from_lbann_tensors(fdir, vocab_file)

orig_file = read_smiles_csv("gt_smiles.txt")
pred_file = read_smiles_csv("pred_smiles.txt")
diff_file = "sd"+"_smiles_metrics.csv"

print("Input/pred SMILES file sizes ", len(orig_file), " ", len(pred_file))

compare_decoded_to_original_smiles(orig_file, pred_file, diff_file)
print("Input/pred SMILES diff file saved to", diff_file)














