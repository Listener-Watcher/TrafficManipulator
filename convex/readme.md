## instructions and case study
Please download the required library including in addition to the libaray mentioned in the main Readme
numpy \\
cvxpy \\
matplotlib \\
sklearn \\

## 0. copy some file to the directory
please copy test.npy and benign_train.npy file from example to this directory for loading data
   ```
   cd AfterImageExtractor/
   python setup.py build_ext --inplace
   ```
## 1.training one SVM
'''
python model.py -M train -mf model.pkl
'''
## 2.generate adversarial examples use convex optimization
python model.py -M gen -mf model.pkl

## 3.evaluate the model
python model.py -M exec -rf rmse.pkl -mf model.pkl
This produces the original RMSE file

## 4. cd .. and back to the main folder
python tools.py this is a step to generate a normalizer.pkl, which is mentioned in the main readme
generated_adversarial_set_path can be the one in convex/mimic_set_convex_norm2.npy or in GAN/mimic_gen_*.npy

python main.py -m example/test.pcap -b generated_adversarial_set_path -n example/normalizer.pkl -i example/init.pcap
execute PSO algorithm

## evaluate the mutated traffic
python eval_svm.py -op example/test.pcap -or convex/rmse.pkl -of example/test.npy -b generated_adversarial_set_path -n example/normalizer.pkl
