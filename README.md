# NLP_Final_Project
The following natural language processing project focuses on exploration of stopword behaviour for chattype messages in three different corpuses.
Main.ipynb contains the main tasks from 1. to 10. research questions and their results for a given corpus. Thus only a portion of results is accecible here but all results will be available in the written paper.


classifierTest.ipynb contains the testing and implementation portion on creating a classifier for the latter two datasets (ClariQ and Ubuntu). (To classify dialogue acts so that analysis can be uniform accross datasets. Training done on nps_chat where dialogue act labels also originate for all sections.) 


Data folder will include the dataset from ClariQ and a portion of Ubuntu corpus (limited due to the massive quantity of it making it impractical to fully analyze). NPS chat corpus can be found through NLTK.


Task 9. will be ran on a seperate notebook named classifierTest.ipynb. It includes the classifier used for our two datasets as well as the state of the art implementation for task 9. utilizing RoBERTa and dialogue GRU model on the NPS chat.
Frozen Distilroberta-base utterance vectors are used in addition with the one trained layer of GRU + classifier. 
Git hub page of the study used for the state of the art solution: https://github.com/declare-lab/dialogue-understanding.


Utilization of AI (co-pilot) and other resources will be commented above the code in question and clarified on their usage.