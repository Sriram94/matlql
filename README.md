# Learning from Multiple Independent Advisors in Multi-agent Reinforcement Learning

Implementation of MA-TLQL and MA-TLAC algorithms for the AAMAS 2023 paper: [Learning from Multiple Independent Advisors in Multi-agent Reinforcement Learning.](https://arxiv.org/pdf/2301.11153.pdf)

 
## Code structure


- See folder playground for Pommerman environment with the MA-TLQL, MA-TLAC, and the baseline algorithms.


- See folder pursuitcode for Pursuit environment with the MA-TLQL, MA-TLAC, and the baseline algorithms.

- See folder mpecode for Predator-Prey environment with the MA-TLAC, and the baseline algorithms.



### In each of directories, the files most relevant to our research are:


- playground/examples/onevsone_differentquality/matlqlvsdqn.py:  This file contains the code for training and testing the neural network implementation of MA-TLQL algorithm vs DQN in the two agent Pommerman version with four different and sufficient advisors.

Similarly other scripts correspond to training and testing of other baselines against DQN. Change the directory to 'onevsone\_similarquality' for the two agent Pommerman version with four similar and sufficient advisors, 'onevsone\_insufficient\_similarquality' for the two agent
Pommerman version with four similar and insufficient advisors, and 'onevsone\_insufficient\_differentquality' for the two agent
Pommerman version with four different and insufficient advisors. 


Similarly, the folder playground/examples/teamcompetition - contains all the scripts for the Pommerman team domain.


- playground/pommerman/agents:  This file contains the code for using the algorithmic implementations to create agents in Pommerman. The code for the rule-based advisors is given in the corresponding scripts of this folder.



- pursuitcode/pettingzoosislpursuitDQN.py :  This file contains the code for training and testing the DQN algorithm in the Pursuit SISL environment.

Similarly other scripts correspond to training and execution for other algorithms in our paper. Remember to train the DQN algorithm before the others (trained DQN is used as the advisor for the others).   


 
- mpecode/pettingzoompesimpletagDQN.py :  This file contains the code for training and testing the DQN algorithm in the Predator-Prey MPE environment.

Similarly other scripts correspond to training and execution for other algorithms in our paper. Remember to train the DQN algorithm before the others (trained DQN is used as the advisor for the others).   



### Algorithmic Implementation 


- playground/example/onevsone_differentquality/RL_brain_matwolevelql.py :  This file contains the code for the algorithmic implementation of MA-TLQL.  

- playground/example/onevsone_differentquality/RL_brain_matwolevelac.py :  This file contains the code for the algorithmic implementation of MA-TLAC. 

- playground/example/onevsone_differentquality/RL_brain_twolevelql.py :  This file contains the code for the algorithmic implementation of TLQL.

- playground/example/onevsone_differentquality/RL_brain_admiraldm.py :  This file contains the code for the algorithmic implementation of ADMIRAL-DM. 

- playground/example/onevsone_differentquality/RL_brain_CHAT.py :  This file contains the code for the algorithmic implementation of CHAT. 

- playground/example/onevsone_differentquality/RL_brain_DQN.py :  This file contains the code for the algorithmic implementation of DQN. 

- playground/example/onevsone_differentquality/DQfD_V3.py :  This file contains the code for the algorithmic implementation of DQfD. 

These files are almost the same across the different folders.
  

## Installation Instructions for Ubuntu 18.04



### Pommerman 

##### Requirements

Atleast 

- `python==3.7.11`


```shell
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.6
```


- `Tkinter`

```shell
sudo apt-get update
sudo apt-get install python3-tk
```


- `tensorflow 2`

```shell
pip install --upgrade pip
pip install tensorflow
```

- `pandas`

```shell
pip install pandas
```
- `matplotlib`

```shell
pip install matplotlib
```

Download the files and store them in a separate directory to install packages from the requirements file. 

```shell
cd playground
pip install -U . 
```


For more help with the installation, look at the instrctions in [Playground](https://github.com/MultiAgentLearning/playground). 

Now you can just run the respective files mentioned in the above section to run our code.


For the Pursuit domain you also need to install petting zoo library. 

### Petting Zoo (Pursuit and Predator-Prey Environments)


##### Requirements

Atleast

- `Gym` (Version 0.18.0)

```shell
pip install gym==0.18.0
```

- `pettingzoo` (Version 1.14.0) 

```shell
cd pettingzoo 
pip install ./
```


Please install petting zoo using the instruction above and not from source (our files contains some differences from the source files). 

Now, you can just run the relevant files mentioned in the above section to run our code. 

## Note

This is research code and will not be actively maintained. Please send an email to ***sriram.subramanian@vectorinstitute.ai*** for questions or comments.


## Code Citations

We would like to cite [Playground](https://github.com/MultiAgentLearning/playground) for code in the playground folder. The files for running Pommerman game are retained from this repository, with only the algorithms implemented from our side as described in the code structure here. 
We would also like to cite [Reinforcement Learning and Tensorflow](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow) from which the structure of our algorithmic implementations and implementations of the baselines were obtained. We thank the [Petting Zoo](https://github.com/PettingZoo-Team/PettingZoo) repository for providing the Pursuit environment. We would also like to thank the [go2sea](https://github.com/go2sea/DQfD) repository on which our DQfD baseline implementation is based. 

## Paper citation

If you found this helpful, please cite the following paper:

<pre>



@InProceedings{Srirammatlql2023,
  title = 	 {Learning from Multiple Independent Advisors in Multi-agent Reinforcement Learning},
  author = 	 {Subramanian, Sriram Ganapathi and Taylor, Matthew E. and Larson, Kate and Crowley, Mark} 
  booktitle = 	 {Proceedings of the International Conference on Autonomous Agents and Multi Agent Systems (AAMAS 2023)},
  year = 	 {2023},
  editor = 	 {A.Ricci, W.Yeoh, N.Admon, B.An},
  address = 	 {London, United Kingdom},
  month = 	 {29 May -- 2 June},
  publisher = 	 {IFAAMAS}
}
</pre>
