# MARL4TA

📄 **Publication**  
We are delighted to share that our work has been accepted and published in  
*Communications in Transportation Research*! 🎉  

Read the paper here:  
[Scalable and reliable multi-agent reinforcement learning for traffic assignment](https://www.sciencedirect.com/science/article/pii/S2772424725000654)  



## Setup Instructions

1. **Create and activate a new Conda environment**:
   ```bash
   conda create -n MARL4TA python=3.8
   conda activate MARL4TA

2. **Install pytorch>=1.9.0 (CUDA>=11.0) manually**

3. **Install required dependencies: You can install all necessary packages by running**:
   ```bash
   pip install gym gymnasium networkx setproctitle pandas matplotlib tensorboardX
   ```
   or
   ```bash
   pip install -r requirements.txt
   ```

4. **Run**:

   ```bash
   python train_ta.py
   ```
