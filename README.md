# credit_card_competition

Purpose Statement: To classify risky credit card users and calculate their respective credit line increase.

 ________________
< Team 8 README  >
 ----------------
        \   ^__^
         \  (oo)\_______
            (__)\       )\/\
                ||----w |
                ||     ||

README
-------------------------------------
For this project we mainly ran everything though google colab, we did not use any obscure python packages and no highly computational complex algorithms were created. The full script only took about 31 seconds, with data preprocessing, variable selection, modeling, and coming up with predictions for all the users. If we were to just run the model by its own, it would decrease the time.

------------
The hardware specifications are:
GPU: 1xTesla K80 , compute 3.7, having 2496 CUDA cores , 12GB GDDR5 VRAM

CPU: 1xsingle core hyper threaded Xeon Processors @2.3Ghz i.e(1 core, 2 threads)

RAM: ~12.6 GB Available

Disk: ~33 GB Available
------------

To run the whole analysis, the 4 .csv files (given at beginning of competition) are needed to be in the working directory. Certain libraries like datetime, sklearn, pandas, numpy, xgb, etc... are needed to be installed




FILES

-------------------------------------
:preprocessing_ID_8.ipynb:

This is the full preprocessing procedure we decided to run on the data. 


-------------------------------------
:modeling_ID_8.ipynb:

This is the full modeling procedure we decided to run on the data.


-------------------------------------
:full_script_ID_8.py: 

This is the combination of preprocessing and modeling into one python script (if preferred to be run in a .py file).

There may be an issue with the current path, the path in the code needs to be changed before it successfully runs.

On google colab's hardware, the script took 31.241 seconds to run.


-------------------------------------
