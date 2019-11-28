# LinearRegression
Batch linear regression using gradient descent

## Command Line Arguments
`python3 LinearReg.py --data data.csv --learningRate 0.0005 --threshold 0.0004`

--data --> name of the csv file<br/>
--learningRate --> the learning rate for the algorithm<br/>
--threshold --> program stops running when the error between two consecutive iterations is less than threshold<br/>

The program is written in such a way that the last column of the data is treated as the target.<br/>
The output file of the program gives the weights and Sum Of Squared Errors <br/>
