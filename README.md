# Social Circles Discovery

## Abstract

Online Social Network (OSN) platforms enable users to organize their social circles for managing their contacts. Organizing social circles is a manual task which is not only tedious but unscalable and in-effective as well. Automation of this process would be beneficial for the users to fulfill their social and professional requirements.
Various approaches have been proposed for the automation of this task but none of these approaches takes edge features like trust level between the nodes alongside with other node and structural features into account. In this project, we propose the Node-Edge K-means clustering algorithm to study the importance of individuals with a good reputation in building social circles. Our approach builds on top of the Genetic Algorithm variant of K-means clustering algorithm that considers only node and structural features for discovering social circles. With the help of intrinsic measures, we then evaluate our approach by comparing the quality of social circles discovered after adding the trust level feature with those discovered without this feature.


## How to run the experiment using python files ?

Following are the two python files, that are needed to run the experiment:

  1. embeddings-generator.py

  2. social-circles-discovery.py

Place both of these files in the same location for ease and then perform the following steps.

1. First we need to generate datasets and prepocess them in order to make them feasible for our approach 
   and for this purpose you need to run embeddings-generator.py file.
   
    a. Install node2vec package before running this script You can install it using following command:
       pip install node2vec or pip3 install node2vec

    b. Then run embeddings-generator.py file using following command:
       python embeddings-generator.py or python3 embeddings-generator.py

    c. After running this script, following two data files are generated:
      
        i. embedded-soc-sign-slashdot
      
        ii. soc-sign-Slashdot090221.txt.gz

2. The dataset files generated in previous step will be used as an input to social-circles-discovery.py file.
   Finally, to run the experiments using our approach you need to run social-circles-discovery.py script using
   following command:
   
    python social-circles-discovery.py or python3 social-circles-discovery.py

    Progress and results of the experiment will be printed on the console.
    
    
## Important Files

1. All the steps performed for social circles discovery and comparison of social circles generated with and without trust feature
   can be seen in jupyter notebook file [social-circles-discovery](https://github.com/muneeb706/social_circles_discovery/blob/main/social-circles-discovery.ipynb).

2. Detailed discussion and analysis can be found in [Discovering social circles in a directed social network using node, structure and edge features](https://github.com/muneeb706/social_circles_discovery/blob/main/Discovering%20social%20circles%20in%20a%20directed%20social%20network%20using%20node%2C%20structure%20and%20edge%20features.pdf).
 
