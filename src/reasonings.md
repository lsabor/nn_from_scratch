  
## Forward pass:  
  
Given X  
A0 = X :: [784,m]  
Z1[10,m] = W1[10,784] * X[784,m] + b1[10]  
A1[10,m] = RelU(Z1[10,m])  
Z2[10,m] = W2[10,10] * A1[10,m] + b2[10]  
Y_hat[10,m] = softmax(A2[10,m])  
Return Y_hat  
  
## Backpropogation:  
  
key:  
VAR{a,b,...} | indicates that VAR has dimensions a x b x ...  
VAR[i,j,...] | indicates the array or value at coordinates (i, j, ...) in VAR  
  
n = number of possible encodings # in general, n can change through a network,  
    but we're assuming that n is used for encodings and also layer size  
m = number of inputs, batch size  
x = number of datapoints per input  
  
(1)     X{x,m} = Input - m examples of x data points  
(2)     Y{n,m} = one-hot encoding of results: e.g. [[0,0,0,1,0,0], ... ] (with m encodings)  
(3)     W1{n,x} = weights applied to X  
(4)     b1{n,1} = biases applied to values going into Z1_n  
(5)     Z1{n,m} = pre-activation function values = W1{n,x} dot X{x,m} + b1{n,1}  
(6)     A1{n,m} = ReLU(Z1{n,m})  
(7)     W2{n,n} = weights applied to A1  
(8)     b2{n,1} = biases applied to values going into Z2_n  
(9)     Z2{n,m} = pre-softmax function values = W2{n,n} dot A1{n,m} + b2{n,1}  
(10)    Y_hat{n,m} = the estimate of Y = softmax(Z2{n,m})  
  
The forward pass looks like:  
X * W1 + b1 = Z1  
ReLU(Z1) = A1  
A1 * W2 + b2 = Z2  
softmax(Z2) = Y_hat  
  
definition: Loss = L{m} = [L_0, L_1, ..., L_m]   
    = - sum over i of Y[i]*ln(Y_hat[i]) = -np.einsum("ij,ij->j",Y, np.log(Y_hat))  
(11)    L{m} = -np.einsum("ij,ij->j",Y, np.log(Y_hat))  
def loss(Y, Y_hat):  
    return -np.einsum("ij,ij->j",Y, np.log(Y_hat))  
We set this loss specifically so that the derivative works out nicely  
  
  
To minimize L, we want to see how L will change with respect to the variables  
that we can control, namely W1, b1, W2, and b2.  
  
To calculate DW2 (dL/dW2), we use the chain rule:  
DW2 = dL/dW2 = dL/dY_hat * dY_hat/dZ2 * dZ2/dW2  
similarly for Db2:  
Db2 = dL/db2 = dL/dY_hat * dY_hat/dZ2 * dZ2/db2  
  
But to calculate DW1 (dL/dW1), it is a little longer  
DW1 = dL/dW1 = dL/dY_hat * dY_hat/dZ2 * dZ2/dA1 * dA1/dZ1 * dZ1/dW1  
similarly for Db1:  
Db1 = dL/db1 = dL/dY_hat * dY_hat/dZ2 * dZ2/dA1 * dA1/dZ1 * dZ1/db1  
  
Now, let's start calculating each of these constituative derivatives.  
  
  
dL/dY_hat:  
dL/dY_hat.shape should be {n,m}  
from (11)    L{m} = -np.dot(Y, np.log(Y_hat)):  
(12)    dL/dY_hat{n,m} = - {sum over i of} (Y[i] / Y_hat[i])  
This shows us the opposite of exactly how Y_hat should change in order to minimize loss   
across the n estimates for each of the m examples.  
  
  
dY_hat/dZ2:  
dY_hat/dZ2.shape should be {n,m}  
from (10)    Y_hat{n,m} = the estimate of Y = softmax(Z2{n,m}):  
given some i,j in range(n) and k,l in range(m):  
Y_hat[i,k] changes with respect to Z2[j,l] only when k == l  
for simplicity, assume k=l and thus drop those terms  
dY_hat[i]/dZ2 has dimension {n}  
dY_hat[i]/dZ2[j] =   
    if i == j --> softmax(Z2[j])*(1-softmax(Z2[j])  
    if i != j --> -softmax(Z2[i])*softmax(Z2[j])  
dY_hat/dZ2 has dimension [n,n] for each entry in m  
dY_hat/dZ2[i,j,k] =  
    if i == j --> softmax(Z2[j,k])*(1-softmax(Z2[j,k])  
    if i != j --> -softmax(Z2[i,k])*softmax(Z2[j,k])  
for simplicity, call p[i, ...] = softmax(Z2[i, ...]). Thus:  
(13)    dY_hat/dZ2[i,j,k]{n,n,m} =  
        if i == j --> p[j,k]*(1-p[j,k])  
        if i != j --> -p[i,k]*p[j,k]  
  
  
DZ2 = dL/dZ2:  
DZ2.shape should be {n,m}  
DZ2 = dL/dY_hat * dY_hat/dZ2  
for now, drop m, so L has dim 1 while Z2 has dim {n}  
let i,j in range(n)  
from (13)   dY_hat/dZ2[i,j,k]{n,n,m} =  
                if i == j --> p[j,k]*(1-p[j,k])  
                if i != j --> -p[i,k]*p[j,k]:  
dL/dZ2[j] = sum over i of dL/dY_hat[i] * dY_hat[i]/dZ2[j]  
    = {when i == j} - Y[j]/Y_hat[j] * Y_hat[j]*(1-Y_hat[j])   
    + {sum over i when i != j of} (- (Y[i] / Y_hat[i]) * -Y_hat[i]*Y_hat[j] )  
    = -Y[j] * (1 - Y_hat[j]) - Y_hat[j] * {sum over i when i != j of} Y[i]  
    = -Y[j] + Y[j] * Y_hat[j] - Y_hat[j] * (-Y[j] + {sum over i of} Y[i]) # added Y[j] into summation  
    = -Y[j] + Y_hat[j] * (-Y[j] - (-Y[j] + 1)) # NOTE: {sum over i of} Y[i] = 1 since   
                                                # Y[i] = 0 for all but 1 i, where it equals 1  
    = -Y[j] + Y_hat[j] * 1 = -Y[j] + Y_hat[j]  
Adding back in k in range(m):  
dL/dZ2[j,k] = -Y[j,k] + Y_hat[j,k]  
(14)    DZ2{n,m} = -Y + Y_hat  
  
  
DW2 = dL/dW2:  
DW2 = dL/dY_hat * dY_hat/dZ2 * dZ2/dW2 = DZ2 * dZ2/dw2  
DW2.shape should be {n,n} (not m because W2 doesn't change across examples)  
finding dZ2/dw2{n,n}:      
from (9)     Z2{n,m} = W2{n,n} dot A1{n,m} + b2{n,1}  
let i,j,k in range(n), dropping m for now  
Z2[i] = W2[i]{n} dot A1{n} + b2[i] = {sum over j} W2[i][j] * A1[j] + b2[i]  
dZ2[i]/dW2[j,k]{1} = 0 if i != j, else A1[k]  
dZ2[i]/dW2[i,k]{1} = A1[k]  
dZ2/dW2{n} = A1  
adding m back in: for l in range(m)  
Z2[i,l]{1} = W2[i] dot A1[l] + b2[i]  
dZ2[i,l]/dW2[i,k]{1} = A1[k,l]  
dZ2[l]/dW2{n} = A1[l]{n}  
dZ2/dW2{n,m} = A1{n,m}  
This shows what you would multiply a delta_W with to get the difference in Z2  
had you added that delta_W to W2 and recalulated Z2 that way  
  
Dropping m again for a moment:  
DW2{n,n} = DZ2 * dZ2/dW2 = DZ2{n} dot A1{n}  
The derivative of the loss with respect to particular values of W2  
To bring m back in the picture, we have to average over all of the losses accrued  
during the training run. Namely m training examples:  
(15)    DW2{n,n} = 1/m * DZ2 * dZ2/dw2 = 1/m * DZ2{n,m} dot A1{n,m}.T{m,n}  
  
  
Db2 = dL/db2:  
Db2 = dL/dY_hat * dY_hat/dZ2 * dZ2/db2 = dZ2 * dZ2/db2  
db2.shape should be {n}   
finding dZ2/db2{n}:  
from (9)     Z2{n,m} = W2{n,n} dot A1{n,m} + b2{n,1}  
let i,j in range(n), dropping m for now  
Z2[i] = W2[i]{n} dot A1{n} + b2[i] = {sum over j} W2[i][j] * A1[j] + b2[i]  
dZ2[i]/db2[j]{1} = 0 if i != j, else 1  
dZ2[i]/db2[i]{1} = 1  
dZ2/db2{n} = 1  
adding m back in: for l in range(m)  
Z2[i,l]{1} = W2[i] dot A1[l] + b2[i]  
dZ2[i,l]/db2[i]{1} = 1  
dZ2[l]/db2{n} = 1{n}  
dZ2/db2{n} = 1{n}  
  
Dropping m again for a moment:  
Db2{n} = DZ2 * dZ2/db2 = DZ2{n} * 1{n} = dZ2{n}  
The derivative of the loss with respect to particular values of b2  
To bring m back in the picture, we have to average over all of the losses accrued  
during the training run. Namely m training examples:  
(16)    Db2{n} = 1/m * DZ2 * dZ2/dw2 = 1/m * 1{n} dot DZ2{n,m} = 1/m * np.sum(DZ2{n,m})  
  
  
DA1 = dL/dA1:  
DA1 = dL/dZ2 * dZ2/dA1  
DA2.shape should be {n,m}  
finding dZ2/dA1:  
from (9)     Z2{n,m} = W2{n,n} dot A1{n,m} + b2{n,1}  
let i,j in range(n), dropping m for now  
Z2[i] = W2[i]{n} dot A1{n} + b2[i] = {sum over j} W2[i][j] * A1[j] + b2[i]  
dZ2[i]/dA1[j] = W2[j,i]  
dZ2/dA1[j] = W2[j]  
dZ2/dA1 = W2  
adding m back in: let k,l in range(m):  
dZ2[:,k]/dA1[:,l] = 0 if l!= k, else W2  
dZ2[:,k]/dA1[:,k] = W2  
dZ2/dA1{n,n} = W2{n,n}  
  
Dropping m again for a moment:  
DA1{n} = DZ2 * dZ2/dA1 = DZ2{n} * W2{n,n} = W2.T{n,n} dot DZ2{n}  
The derivative of loss with respect to a particular A1 value  
Bringing m back in the picture is easy:  
{17}    DA1{n,m} = W2.T{n,n} dot DZ2{n,m}  
  
  
DZ1 = DL/dZ1:  
DZ1 = dL/dA1 * dA1/dZ1  
DZ1.shape should be {n,m}  
finding dA1/dZ1:  
from (6)     A1{n,m} = ReLU(Z1{n,m}):  
ReLU is applied item-wize on Z1, so the process is simple  
dA1/dZ1{n,m} = ReLU_deriv(Z1{n,m})  
(18)    DZ1{n,m} = DA1 * dA1/dZ1 = DA1{n,m} * ReLU_deriv(Z1{n,m})  
We are done already, and note "*" is multipliaction item-wize in this formula  
  
  
DW1 & Db1:  
process is identical to above. Thus:  
(19)    DW1{n,x} = 1/m * DZ2{n,m} dot X{x,m}.T{m,x}  
(20)    Db1{n} = 1/m * np.sum(DZ1{n,m})  
  
  
Now we are done!  
  
  
"""  
