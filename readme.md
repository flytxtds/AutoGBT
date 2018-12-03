# AutoGBT
AutoGBT stands for Automatically Optimized Gradient Boosting Trees, and is used for AutoML in a lifelong machine learning setting to classify large volume high cardinality data streams under concept-drift. AutoGBT was developed by a joint team ('autodidact.ai') from Flytxt, Indian Institute of Technology Delhi and CSIR-CEERI as a part of NIPS 2018 AutoML Challenge (The 3rd AutoML Challenge: AutoML for Lifelong Machine Learning). Our team won the first prize in the challenge. More details of the challenge is available at https://www.4paradigm.com/competition/nips2018. The work will be presented at NIPS 2018 during the Competition Track session (https://nips.cc/Conferences/2018/Schedule?showEvent=10945)

Team:\
1.Jobin Wilson (jobin.wilson@flytxt.com)\
2.Amit Kumar Meher (amit.meher@flytxt.com)\
3.Bivin Vinodkumar Bindu (bivin.vinod@flytxt.com)\
4.Manoj Sharma (mksnith@gmail.com)\
5.Vishakha Pareek (vishakhapareek@ceeri.res.in)

# How to Run
Download the starter kit from the NIPS AutoML from competion webpage (https://competitions.codalab.org/competitions/20203#participate-get_starting_kit) and setup locally as instructed in the readme file within the starter kit. Copy the folder "AutoGBT" into the starting_k folder inside the starter kit. Install docker from https://docs.docker.com/get-started/ and issue the following command to invoke the docker image corresponding to python3 bundle for the challenge.\
docker run -it -u root -v $(pwd):/app/codalab codalab/codalab-legacy:py3 bash

For ingestion, use the following command from the docker shell prompt

python3 AutoML3_ingestion_program/ingestion.py AutoML3_sample_data AutoML3_sample_predictions AutoML3_sample_ref AutoML3_ingestion_program AutoGBT\

For scoring, use the following command from the docker shell prompt\
python3 AutoML3_scoring_program/score.py 'AutoML3_sample_data/*/' AutoML3_sample_predictions AutoML3_scoring_output

