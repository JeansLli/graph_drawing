# Drawing graph with minimum edge crossings


## Force Directed Method
If you want to initialize the graph first, please run
```
python force_directed.py test-5.txt
```
Please install [NetworkX](https://networkx.org/) first.

Then run the evolutionary algorithm.

## Evolutionary Algorithm
The command is like:
```
python EA.py file_name iterations
```
For example:
```
python EA.py test-1.txt 1000
```



## Visualization
To visualize the input and output data, please run
```
python visualization.py test-1.txt
```
The input data is in the fold `instances`, and the output data is in the fold `output_final`

