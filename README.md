# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

# Parallel Analytics Diagnostics

## MAP

Parallel Accelerator Optimizing: Function tensor_map.<locals>._map, minitorch\fast_ops.py (163)

Parallel loop listing for Function tensor_map.<locals>._map, minitorch\fast_ops.py (163)
-------------------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                                    |
        out: Storage,                                                                            |
        out_shape: Shape,                                                                        |
        out_strides: Strides,                                                                    |
        in_storage: Storage,                                                                     |
        in_shape: Shape,                                                                         |
        in_strides: Strides,                                                                     |
    ) -> None:                                                                                   |
                                                                                                 |
        if (np.array_equal(out_strides, in_strides) and np.array_equal(out_shape, in_shape)):    |
            for i in prange(len(out)):-----------------------------------------------------------| #0
                out[i] = fn(in_storage[i])                                                       |
                                                                                                 |
        else:                                                                                    |
            for i in prange(len(out)):-----------------------------------------------------------| #1
                out_index: Index = np.empty_like(out_shape, dtype=np.int32)                      |
                in_index: Index = np.empty_like(in_shape, dtype=np.int32)                        |
                to_index(i, out_shape, out_index)                                                |
                broadcast_index(out_index, out_shape, in_shape, in_index)                        |
                                                                                                 |
                in_position = index_to_position(in_index, in_strides)                            |
                out[i] = fn(in_storage[in_position])                                             |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #0, #1).

----------------------------- Before Optimisation ------------------------------

------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.


---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None

## ZIP

Parallel Accelerator Optimizing: Function tensor_zip.<locals>._zip, minitorch\fast_ops.py (212)

Parallel loop listing for Function tensor_zip.<locals>._zip, minitorch\fast_ops.py (212)
-----------------------------------------------------------------------------|loop #ID
    def _zip(                                                                |
        out: Storage,                                                        |
        out_shape: Shape,                                                    |
        out_strides: Strides,                                                |
        a_storage: Storage,                                                  |
        a_shape: Shape,                                                      |
        a_strides: Strides,                                                  |
        b_storage: Storage,                                                  |
        b_shape: Shape,                                                      |
        b_strides: Strides,                                                  |
    ) -> None:                                                               |
                                                                             |
        if (                                                                 |
            np.array_equal(out_strides, a_strides)                           |
            and np.array_equal(out_shape, a_shape)                           |
            and np.array_equal(out_strides, b_strides)                       |
            and np.array_equal(out_shape, b_shape)                           |
        ):                                                                   |
            for i in prange(len(out)):---------------------------------------| #2
                out[i] = fn(a_storage[i], b_storage[i])                      |
                                                                             |
        else:                                                                |
            for i in prange(len(out)):---------------------------------------| #3
                out_index = np.empty_like(out_shape, dtype=np.int32)         |
                a_index = np.empty_like(a_shape, dtype=np.int32)             |
                b_index = np.empty_like(b_shape, dtype=np.int32)             |
                to_index(i, out_shape, out_index)                            |
                broadcast_index(out_index, out_shape, a_shape, a_index)      |
                broadcast_index(out_index, out_shape, b_shape, b_index)      |
                                                                             |
                a_position = index_to_position(a_index, a_strides)           |
                b_position = index_to_position(b_index, b_strides)           |
                out[i] = fn(a_storage[a_position], b_storage[b_position])    |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #2, #3).

----------------------------- Before Optimisation ------------------------------

------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.


---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None

## REDUCE

Parallel Accelerator Optimizing: Function tensor_reduce.<locals>._reduce, minitorch\fast_ops.py (270)

Parallel loop listing for Function tensor_reduce.<locals>._reduce, minitorch\fast_ops.py (270)
------------------------------------------------------------------------------------------------------|loop #ID
    def _reduce(                                                                                      |
        out: Storage,                                                                                 |
        out_shape: Shape,                                                                             |
        out_strides: Strides,                                                                         |
        a_storage: Storage,                                                                           |
        a_shape: Shape,                                                                               |
        a_strides: Strides,                                                                           |
        reduce_dim: int,                                                                              |
    ) -> None:                                                                                        |
                                                                                                      |
        for i in prange(len(out)):--------------------------------------------------------------------| #4
            out_index = np.empty_like(out_shape, dtype=np.int32)                                      |
            to_index(i, out_shape, out_index)                                                         |
            a_position = index_to_position(out_index, a_strides)                                      |
                                                                                                      |
            reduce_value = out[i]                                                                     |
            for j in range(a_shape[reduce_dim]):                                                      |
                reduce_value = fn(reduce_value, a_storage[a_position + j * a_strides[reduce_dim]])    |
                                                                                                      |
            out[i] = reduce_value                                                                     |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #4).

----------------------------- Before Optimisation ------------------------------

------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.


---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None

## MATRIX MULTIPLY

Parallel Accelerator Optimizing: Function _tensor_matrix_multiply, minitorch\fast_ops.py (294)

Parallel loop listing for Function _tensor_matrix_multiply, minitorch\fast_ops.py (294)
-------------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                     |
    out: Storage,                                                                                |
    out_shape: Shape,                                                                            |
    out_strides: Strides,                                                                        |
    a_storage: Storage,                                                                          |
    a_shape: Shape,                                                                              |
    a_strides: Strides,                                                                          |
    b_storage: Storage,                                                                          |
    b_shape: Shape,                                                                              |
    b_strides: Strides,                                                                          |
) -> None:                                                                                       |
    """NUMBA tensor matrix multiply function.                                                    |
                                                                                                 |
    Should work for any tensor shapes that broadcast as long as                                  |
                                                                                                 |
    ```assert a_shape[-1] == b_shape[-2]```                                                                                          |
                                                                                                 |
    Optimizations:                                                                               |
                                                                                                 |
    * Outer loop in parallel                                                                     |
    * No index buffers or function calls                                                         |
    * Inner loop should have no global writes, 1 multiply.                                       |
                                                                                                 |
                                                                                                 |
    Args:                                                                                        |
    ----                                                                                         |
        out (Storage): storage for `out` tensor                                                  |
        out_shape (Shape): shape for `out` tensor                                                |
        out_strides (Strides): strides for `out` tensor                                          |
        a_storage (Storage): storage for `a` tensor                                              |
        a_shape (Shape): shape for `a` tensor                                                    |
        a_strides (Strides): strides for `a` tensor                                              |
        b_storage (Storage): storage for `b` tensor                                              |
        b_shape (Shape): shape for `b` tensor                                                    |
        b_strides (Strides): strides for `b` tensor                                              |
                                                                                                 |
    Returns:                                                                                     |
    -------                                                                                      |
        None : Fills in `out`                                                                    |
                                                                                                 |
    """                                                                                          |
    a_batch_stride = a_strides if a_shape > 1 else 0                                             |
    b_batch_stride = b_strides if b_shape > 1 else 0                                             |
                                                                                                 |
    K = a_shape[-1] # must be equal to b_shape[-2]                                               |
    D, R, C = out_shape[:3]                                                                      |
                                                                                                 |
    # out[d, r, c] = sum_k a[d_a, r, k] * b[d_b, k, c]                                           |
    for d in prange(D):--------------------------------------------------------------------------| #5
        for r in range(R):                                                                       |
            for c in range(C):                                                                   |
                out_position = (                                                                 |
                    d * out_strides +                                                            |
                    r * out_strides[1] +                                                         |
                    c * out_strides[2]                                                           |
                )                                                                                |
                                                                                                 |
                dot_product = 0.0                                                                |
                for k in range(K):                                                               |
                    dot_product += (                                                             |
                        a_storage[d * a_batch_stride + r * a_strides[1] + k * a_strides[2]] *    |
                        b_storage[d * b_batch_stride + k * b_strides[1] + c * b_strides[2]]      |
                    )                                                                            |
                                                                                                 |
                out[out_position] = dot_product                                                  |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #5).

----------------------------- Before Optimisation ------------------------------

------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.


---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None

# Performance Comparison: CPU vs GPU Operations

<img src="https://github.com/Cornell-Tech-ML/mod3-Feifenyx/blob/master/timing.png">

# Model Training Results

## Xor Dataset

### CPU Training

#### Hyperparameters

* Size of Hidden Layer: 100
* Learning Rate: 0.05
* Number of Epochs: 490

#### Output Record

* Training Time per Epoch: 0.1590s
* Final Accuracy: 50/50 = 1.0

#### Output Training Log

* Epoch 0  loss 5.901636  correct 28  time 18.1152s
* Epoch 10  loss 6.525591  correct 37  time 1.8176s
* Epoch 20  loss 5.254355  correct 45  time 1.0044s
* Epoch 30  loss 3.380746  correct 45  time 0.7158s
* Epoch 40  loss 5.293480  correct 46  time 0.5684s
* Epoch 50  loss 3.130665  correct 46  time 0.4794s
* Epoch 60  loss 2.017939  correct 45  time 0.4200s
* Epoch 70  loss 2.210279  correct 46  time 0.3765s
* Epoch 80  loss 2.485437  correct 47  time 0.3451s
* Epoch 90  loss 1.264647  correct 47  time 0.3194s
* Epoch 100  loss 1.888201  correct 47  time 0.3027s
* Epoch 110  loss 3.158654  correct 46  time 0.2918s
* Epoch 120  loss 1.575288  correct 47  time 0.2768s
* Epoch 130  loss 1.178675  correct 47  time 0.2641s
* Epoch 140  loss 2.745693  correct 47  time 0.2532s
* Epoch 150  loss 0.640090  correct 47  time 0.2437s
* Epoch 160  loss 2.670526  correct 47  time 0.2355s
* Epoch 170  loss 2.283714  correct 49  time 0.2283s
* Epoch 180  loss 1.856469  correct 48  time 0.2218s
* Epoch 190  loss 2.150122  correct 48  time 0.2160s
* Epoch 200  loss 1.735809  correct 49  time 0.2115s
* Epoch 210  loss 1.074142  correct 49  time 0.2118s
* Epoch 220  loss 1.275779  correct 49  time 0.2073s
* Epoch 230  loss 1.197815  correct 48  time 0.2031s
* Epoch 240  loss 2.025413  correct 49  time 0.1993s
* Epoch 250  loss 0.603523  correct 49  time 0.1957s
* Epoch 260  loss 0.603807  correct 49  time 0.1925s
* Epoch 270  loss 0.782228  correct 50  time 0.1894s
* Epoch 280  loss 0.918185  correct 49  time 0.1866s
* Epoch 290  loss 2.300648  correct 50  time 0.1840s
* Epoch 300  loss 0.373037  correct 50  time 0.1817s
* Epoch 310  loss 0.383877  correct 50  time 0.1821s
* Epoch 320  loss 0.075094  correct 50  time 0.1807s
* Epoch 330  loss 0.564582  correct 50  time 0.1787s
* Epoch 340  loss 0.812634  correct 49  time 0.1767s
* Epoch 350  loss 0.869848  correct 50  time 0.1748s
* Epoch 360  loss 1.707782  correct 49  time 0.1730s
* Epoch 370  loss 0.091347  correct 50  time 0.1713s
* Epoch 380  loss 0.568894  correct 50  time 0.1698s
* Epoch 390  loss 0.495990  correct 50  time 0.1682s
* Epoch 400  loss 0.733934  correct 50  time 0.1668s
* Epoch 410  loss 2.104654  correct 46  time 0.1670s
* Epoch 420  loss 0.671352  correct 50  time 0.1670s
* Epoch 430  loss 0.727687  correct 50  time 0.1657s
* Epoch 440  loss 0.604316  correct 50  time 0.1645s
* Epoch 450  loss 0.855598  correct 50  time 0.1633s
* Epoch 460  loss 0.341715  correct 50  time 0.1621s
* Epoch 470  loss 0.424476  correct 50  time 0.1611s
* Epoch 480  loss 0.681700  correct 50  time 0.1600s
* Epoch 490  loss 0.156606  correct 50  time 0.1590s

### GPU Training

#### Hyperparameters

* Size of Hidden Layer: 100
* Learning Rate: 0.05
* Number of Epochs: 490

#### Output Record

* Training Time per Epoch: 1.6940s
* Final Accuracy: 50/50 = 1.0

#### Output Training Log

* Epoch 0  loss 6.118058  correct 31  time 4.7257s
* Epoch 10  loss 3.915026  correct 46  time 2.0051s
* Epoch 20  loss 3.820226  correct 43  time 1.8370s
* Epoch 30  loss 3.261791  correct 48  time 1.7789s
* Epoch 40  loss 1.637724  correct 47  time 1.7659s
* Epoch 50  loss 1.591286  correct 46  time 1.7446s
* Epoch 60  loss 1.834990  correct 47  time 1.7327s
* Epoch 70  loss 1.516950  correct 48  time 1.7290s
* Epoch 80  loss 1.274214  correct 48  time 1.7192s
* Epoch 90  loss 3.757446  correct 47  time 1.7198s
* Epoch 100  loss 2.409769  correct 45  time 1.7161s
* Epoch 110  loss 2.059908  correct 48  time 1.7097s
* Epoch 120  loss 1.383513  correct 48  time 1.7105s
* Epoch 130  loss 1.406557  correct 49  time 1.7079s
* Epoch 140  loss 0.610072  correct 49  time 1.7089s
* Epoch 150  loss 1.581185  correct 49  time 1.7104s
* Epoch 160  loss 0.668178  correct 48  time 1.7065s
* Epoch 170  loss 0.885359  correct 49  time 1.7067s
* Epoch 180  loss 0.622766  correct 49  time 1.7048s
* Epoch 190  loss 1.299796  correct 47  time 1.7022s
* Epoch 200  loss 0.786597  correct 50  time 1.7040s
* Epoch 210  loss 0.741361  correct 50  time 1.7017s
* Epoch 220  loss 0.701408  correct 50  time 1.6996s
* Epoch 230  loss 0.487478  correct 50  time 1.7008s
* Epoch 240  loss 0.530209  correct 50  time 1.6988s
* Epoch 250  loss 0.403152  correct 50  time 1.6973s
* Epoch 260  loss 0.761734  correct 50  time 1.6980s
* Epoch 270  loss 0.674985  correct 50  time 1.6994s
* Epoch 280  loss 0.508984  correct 50  time 1.7005s
* Epoch 290  loss 0.436447  correct 50  time 1.6987s
* Epoch 300  loss 0.792291  correct 50  time 1.6972s
* Epoch 310  loss 0.617751  correct 50  time 1.6986s
* Epoch 320  loss 0.237147  correct 50  time 1.6971s
* Epoch 330  loss 0.763980  correct 50  time 1.6958s
* Epoch 340  loss 0.458553  correct 50  time 1.6967s
* Epoch 350  loss 0.025941  correct 50  time 1.6955s
* Epoch 360  loss 0.340206  correct 50  time 1.6949s
* Epoch 370  loss 0.548331  correct 50  time 1.6949s
* Epoch 380  loss 0.268415  correct 50  time 1.6936s
* Epoch 390  loss 0.320643  correct 50  time 1.6947s
* Epoch 400  loss 0.360275  correct 50  time 1.6937s
* Epoch 410  loss 0.555782  correct 50  time 1.6949s
* Epoch 420  loss 0.450572  correct 50  time 1.6959s
* Epoch 430  loss 0.447556  correct 50  time 1.6949s
* Epoch 440  loss 0.448519  correct 50  time 1.6943s
* Epoch 450  loss 0.260122  correct 50  time 1.6950s
* Epoch 460  loss 0.626666  correct 50  time 1.6944s
* Epoch 470  loss 0.234632  correct 50  time 1.6953s
* Epoch 480  loss 0.598414  correct 50  time 1.6945s
* Epoch 490  loss 0.232453  correct 50  time 1.6940s

## Simple Dataset

### CPU Training

#### Hyperparameters

* Size of Hidden Layer: 100
* Learning Rate: 0.05
* Number of Epochs: 490

#### Output Record

* Training Time per Epoch: 0.1578s
* Final Accuracy: 50/50 = 1.0

#### Output Training Log

* Epoch 0  loss 5.936367  correct 43  time 17.2922s
* Epoch 10  loss 3.140031  correct 49  time 1.6726s
* Epoch 20  loss 1.284873  correct 48  time 0.9296s
* Epoch 30  loss 0.248656  correct 48  time 0.6664s
* Epoch 40  loss 1.537836  correct 50  time 0.5333s
* Epoch 50  loss 0.779840  correct 49  time 0.4505s
* Epoch 60  loss 0.223771  correct 50  time 0.3959s
* Epoch 70  loss 0.629654  correct 50  time 0.3684s
* Epoch 80  loss 0.831893  correct 50  time 0.3396s
* Epoch 90  loss 0.194828  correct 50  time 0.3143s
* Epoch 100  loss 0.985805  correct 50  time 0.2940s
* Epoch 110  loss 0.584488  correct 50  time 0.2774s
* Epoch 120  loss 0.406154  correct 50  time 0.2637s
* Epoch 130  loss 0.103237  correct 50  time 0.2521s
* Epoch 140  loss 0.600637  correct 50  time 0.2421s
* Epoch 150  loss 0.378023  correct 50  time 0.2332s
* Epoch 160  loss 0.051790  correct 49  time 0.2255s
* Epoch 170  loss 0.045167  correct 50  time 0.2229s
* Epoch 180  loss 0.252998  correct 50  time 0.2192s
* Epoch 190  loss 0.013520  correct 50  time 0.2135s
* Epoch 200  loss 0.461849  correct 50  time 0.2084s
* Epoch 210  loss 0.158033  correct 50  time 0.2037s
* Epoch 220  loss 1.104596  correct 50  time 0.1996s
* Epoch 230  loss 0.543609  correct 50  time 0.1957s
* Epoch 240  loss 0.278672  correct 50  time 0.1922s
* Epoch 250  loss 0.245958  correct 50  time 0.1889s
* Epoch 260  loss 0.084013  correct 50  time 0.1859s
* Epoch 270  loss 0.044117  correct 50  time 0.1846s
* Epoch 280  loss 0.237453  correct 50  time 0.1850s
* Epoch 290  loss 0.008399  correct 50  time 0.1825s
* Epoch 300  loss 0.013612  correct 50  time 0.1801s
* Epoch 310  loss 0.506297  correct 50  time 0.1778s
* Epoch 320  loss 0.296567  correct 50  time 0.1757s
* Epoch 330  loss 0.000428  correct 50  time 0.1738s
* Epoch 340  loss 0.115224  correct 50  time 0.1719s
* Epoch 350  loss 0.197337  correct 50  time 0.1701s
* Epoch 360  loss 0.061106  correct 50  time 0.1685s
* Epoch 370  loss 0.246288  correct 50  time 0.1677s
* Epoch 380  loss 0.480326  correct 50  time 0.1686s
* Epoch 390  loss 0.198131  correct 50  time 0.1671s
* Epoch 400  loss 0.908010  correct 50  time 0.1657s
* Epoch 410  loss 0.556973  correct 50  time 0.1643s
* Epoch 420  loss 0.052237  correct 50  time 0.1630s
* Epoch 430  loss 0.005176  correct 50  time 0.1618s
* Epoch 440  loss 0.425322  correct 50  time 0.1606s
* Epoch 450  loss 0.177765  correct 50  time 0.1595s
* Epoch 460  loss 0.619864  correct 50  time 0.1584s
* Epoch 470  loss 0.010896  correct 50  time 0.1574s
* Epoch 480  loss 0.171451  correct 50  time 0.1583s
* Epoch 490  loss 0.003659  correct 50  time 0.1578s

### GPU Training

#### Hyperparameters

* Size of Hidden Layer: 100
* Learning Rate: 0.05
* Number of Epochs: 490

#### Output Record

* Training Time per Epoch: 1.7121s
* Final Accuracy: 50/50 = 1.0

#### Output Training Log

* Epoch 0  loss 5.642910  correct 41  time 4.6637s
* Epoch 10  loss 1.551665  correct 50  time 1.9693s
* Epoch 20  loss 0.790168  correct 50  time 1.8598s
* Epoch 30  loss 0.620634  correct 50  time 1.8046s
* Epoch 40  loss 0.518381  correct 50  time 1.7908s
* Epoch 50  loss 0.309143  correct 50  time 1.7856s
* Epoch 60  loss 1.289739  correct 50  time 1.7702s
* Epoch 70  loss 0.794980  correct 50  time 1.7677s
* Epoch 80  loss 0.734178  correct 50  time 1.7568s
* Epoch 90  loss 0.917564  correct 50  time 1.7556s
* Epoch 100  loss 0.896275  correct 50  time 1.7476s
* Epoch 110  loss 0.413545  correct 50  time 1.7400s
* Epoch 120  loss 0.235732  correct 50  time 1.7415s
* Epoch 130  loss 0.606734  correct 50  time 1.7365s
* Epoch 140  loss 0.166968  correct 50  time 1.7359s
* Epoch 150  loss 0.269185  correct 50  time 1.7337s
* Epoch 160  loss 1.046914  correct 50  time 1.7297s
* Epoch 170  loss 0.253347  correct 50  time 1.7318s
* Epoch 180  loss 0.222117  correct 50  time 1.7308s
* Epoch 190  loss 0.001984  correct 50  time 1.7321s
* Epoch 200  loss 0.160428  correct 50  time 1.7310s
* Epoch 210  loss 0.433505  correct 50  time 1.7282s
* Epoch 220  loss 0.501898  correct 50  time 1.7290s
* Epoch 230  loss 0.159269  correct 50  time 1.7265s
* Epoch 240  loss 0.332610  correct 50  time 1.7238s
* Epoch 250  loss 0.398341  correct 50  time 1.7243s
* Epoch 260  loss 0.002424  correct 50  time 1.7221s
* Epoch 270  loss 0.101966  correct 50  time 1.7220s
* Epoch 280  loss 0.041931  correct 50  time 1.7205s
* Epoch 290  loss 0.050622  correct 50  time 1.7183s
* Epoch 300  loss 0.274705  correct 50  time 1.7194s
* Epoch 310  loss 0.405683  correct 50  time 1.7179s
* Epoch 320  loss 0.051389  correct 50  time 1.7168s
* Epoch 330  loss 0.063292  correct 50  time 1.7203s
* Epoch 340  loss 0.130058  correct 50  time 1.7184s
* Epoch 350  loss 0.022290  correct 50  time 1.7191s
* Epoch 360  loss 0.020259  correct 50  time 1.7172s
* Epoch 370  loss 0.264871  correct 50  time 1.7160s
* Epoch 380  loss 0.051773  correct 50  time 1.7164s
* Epoch 390  loss 0.276120  correct 50  time 1.7150s
* Epoch 400  loss 0.255637  correct 50  time 1.7137s
* Epoch 410  loss 0.076448  correct 50  time 1.7143s
* Epoch 420  loss 0.037527  correct 50  time 1.7131s
* Epoch 430  loss 0.035664  correct 50  time 1.7132s
* Epoch 440  loss 0.188383  correct 50  time 1.7125s
* Epoch 450  loss 0.196944  correct 50  time 1.7114s
* Epoch 460  loss 0.040619  correct 50  time 1.7119s
* Epoch 470  loss 0.006619  correct 50  time 1.7125s
* Epoch 480  loss 0.179202  correct 50  time 1.7121s
* Epoch 490  loss 0.036096  correct 50  time 1.7121s

## Split Dataset

### CPU Training

#### Hyperparameters

* Size of Hidden Layer: 100
* Learning Rate: 0.05
* Number of Epochs: 490

#### Output Record

* Training Time per Epoch: 0.1611s
* Final Accuracy: 50/50 = 1.0

#### Output Training Log

* Epoch 0  loss 4.627439  correct 31  time 17.5682s
* Epoch 10  loss 6.293920  correct 40  time 1.6981s
* Epoch 20  loss 4.858876  correct 40  time 0.9423s
* Epoch 30  loss 3.449796  correct 39  time 0.6739s
* Epoch 40  loss 4.473870  correct 43  time 0.5366s
* Epoch 50  loss 2.768549  correct 47  time 0.4532s
* Epoch 60  loss 1.048215  correct 46  time 0.3984s
* Epoch 70  loss 3.107282  correct 49  time 0.3616s
* Epoch 80  loss 1.659602  correct 48  time 0.3413s
* Epoch 90  loss 1.450582  correct 49  time 0.3161s
* Epoch 100  loss 1.044284  correct 47  time 0.2980s
* Epoch 110  loss 2.379893  correct 49  time 0.2819s
* Epoch 120  loss 1.815596  correct 48  time 0.2678s
* Epoch 130  loss 0.430882  correct 47  time 0.2557s
* Epoch 140  loss 0.734493  correct 50  time 0.2455s
* Epoch 150  loss 1.767973  correct 50  time 0.2365s
* Epoch 160  loss 0.427268  correct 50  time 0.2287s
* Epoch 170  loss 0.299726  correct 49  time 0.2233s
* Epoch 180  loss 1.647994  correct 48  time 0.2223s
* Epoch 190  loss 0.394462  correct 49  time 0.2165s
* Epoch 200  loss 0.236041  correct 49  time 0.2113s
* Epoch 210  loss 0.723624  correct 49  time 0.2065s
* Epoch 220  loss 0.788124  correct 49  time 0.2021s
* Epoch 230  loss 0.504174  correct 49  time 0.1982s
* Epoch 240  loss 0.780513  correct 50  time 0.1946s
* Epoch 250  loss 0.110878  correct 50  time 0.1912s
* Epoch 260  loss 0.896459  correct 49  time 0.1882s
* Epoch 270  loss 0.076044  correct 50  time 0.1854s
* Epoch 280  loss 0.815799  correct 50  time 0.1858s
* Epoch 290  loss 0.462476  correct 50  time 0.1840s
* Epoch 300  loss 0.349573  correct 50  time 0.1816s
* Epoch 310  loss 0.650310  correct 49  time 0.1793s
* Epoch 320  loss 0.800631  correct 50  time 0.1771s
* Epoch 330  loss 0.408306  correct 50  time 0.1751s
* Epoch 340  loss 0.716351  correct 50  time 0.1732s
* Epoch 350  loss 0.114832  correct 50  time 0.1715s
* Epoch 360  loss 0.022259  correct 50  time 0.1698s
* Epoch 370  loss 0.720400  correct 50  time 0.1682s
* Epoch 380  loss 0.308532  correct 50  time 0.1687s
* Epoch 390  loss 0.449633  correct 50  time 0.1681s
* Epoch 400  loss 0.792755  correct 50  time 0.1667s
* Epoch 410  loss 0.492824  correct 50  time 0.1653s
* Epoch 420  loss 0.736654  correct 50  time 0.1640s
* Epoch 430  loss 0.095913  correct 50  time 0.1628s
* Epoch 440  loss 0.030700  correct 50  time 0.1616s
* Epoch 450  loss 0.674638  correct 50  time 0.1605s
* Epoch 460  loss 0.706095  correct 50  time 0.1594s
* Epoch 470  loss 0.679198  correct 50  time 0.1593s
* Epoch 480  loss 0.228174  correct 50  time 0.1605s
* Epoch 490  loss 0.341356  correct 50  time 0.1611s

### GPU Training

#### Hyperparameters

* Size of Hidden Layer: 100
* Learning Rate: 0.05
* Number of Epochs: 490

#### Output Record

* Training Time per Epoch: 1.6960s
* Final Accuracy: 49/50 = 0.98

#### Output Training Log

* Epoch 0  loss 6.339119  correct 29  time 5.2268s
* Epoch 10  loss 7.952821  correct 40  time 1.9880s
* Epoch 20  loss 4.606590  correct 44  time 1.8315s
* Epoch 30  loss 7.378894  correct 46  time 1.8011s
* Epoch 40  loss 3.890324  correct 45  time 1.7644s
* Epoch 50  loss 2.971108  correct 48  time 1.7529s
* Epoch 60  loss 0.957482  correct 48  time 1.7471s
* Epoch 70  loss 1.496808  correct 49  time 1.7338s
* Epoch 80  loss 0.801458  correct 49  time 1.7366s
* Epoch 90  loss 2.291786  correct 49  time 1.7281s
* Epoch 100  loss 1.599642  correct 49  time 1.7216s
* Epoch 110  loss 1.977205  correct 49  time 1.7300s
* Epoch 120  loss 2.081597  correct 49  time 1.7237s
* Epoch 130  loss 0.884337  correct 49  time 1.7215s
* Epoch 140  loss 1.252569  correct 49  time 1.7199s
* Epoch 150  loss 0.956040  correct 49  time 1.7162s
* Epoch 160  loss 1.196486  correct 50  time 1.7164s
* Epoch 170  loss 0.305142  correct 49  time 1.7129s
* Epoch 180  loss 1.104644  correct 49  time 1.7097s
* Epoch 190  loss 1.097359  correct 49  time 1.7120s
* Epoch 200  loss 0.832257  correct 49  time 1.7093s
* Epoch 210  loss 0.796570  correct 50  time 1.7074s
* Epoch 220  loss 0.330407  correct 50  time 1.7079s
* Epoch 230  loss 0.465978  correct 49  time 1.7052s
* Epoch 240  loss 0.321900  correct 50  time 1.7056s
* Epoch 250  loss 0.583507  correct 50  time 1.7075s
* Epoch 260  loss 0.978363  correct 49  time 1.7051s
* Epoch 270  loss 0.582742  correct 49  time 1.7059s
* Epoch 280  loss 0.515169  correct 50  time 1.7039s
* Epoch 290  loss 0.186555  correct 49  time 1.7022s
* Epoch 300  loss 0.290428  correct 49  time 1.7032s
* Epoch 310  loss 0.226122  correct 50  time 1.7013s
* Epoch 320  loss 0.238078  correct 50  time 1.7005s
* Epoch 330  loss 0.364340  correct 49  time 1.7006s
* Epoch 340  loss 1.312201  correct 49  time 1.6991s
* Epoch 350  loss 0.157038  correct 50  time 1.7001s
* Epoch 360  loss 1.394613  correct 49  time 1.6986s
* Epoch 370  loss 0.035380  correct 50  time 1.6977s
* Epoch 380  loss 0.610708  correct 50  time 1.6988s
* Epoch 390  loss 0.081749  correct 49  time 1.6975s
* Epoch 400  loss 0.226278  correct 50  time 1.6987s
* Epoch 410  loss 0.082480  correct 50  time 1.6994s
* Epoch 420  loss 0.441329  correct 49  time 1.6982s
* Epoch 430  loss 1.463729  correct 49  time 1.6978s
* Epoch 440  loss 0.253184  correct 49  time 1.6979s
* Epoch 450  loss 0.266758  correct 49  time 1.6967s
* Epoch 460  loss 0.128809  correct 49  time 1.6973s
* Epoch 470  loss 1.001542  correct 49  time 1.6964s
* Epoch 480  loss 0.287266  correct 49  time 1.6953s
* Epoch 490  loss 0.848460  correct 49  time 1.6960s

## Split Dataset (Bigger Model)

### CPU Training

#### Hyperparameters

* Size of Hidden Layer: 200
* Learning Rate: 0.05
* Number of Epochs: 490

#### Output Record

* Training Time per Epoch: 0.3218s
* Final Accuracy: 49/50 = 0.98

#### Output Training Log

* Epoch 0  loss 28.569403  correct 32  time 17.6293s
* Epoch 10  loss 5.091883  correct 42  time 1.8376s
* Epoch 20  loss 3.212468  correct 44  time 1.0849s
* Epoch 30  loss 6.604867  correct 42  time 0.8177s
* Epoch 40  loss 4.526864  correct 43  time 0.7087s
* Epoch 50  loss 2.318740  correct 46  time 0.6211s
* Epoch 60  loss 3.051614  correct 48  time 0.5642s
* Epoch 70  loss 1.544548  correct 43  time 0.5209s
* Epoch 80  loss 3.116865  correct 47  time 0.4951s
* Epoch 90  loss 2.183722  correct 47  time 0.4759s
* Epoch 100  loss 3.068123  correct 47  time 0.4542s
* Epoch 110  loss 4.026271  correct 46  time 0.4363s
* Epoch 120  loss 2.146653  correct 48  time 0.4215s
* Epoch 130  loss 3.130038  correct 44  time 0.4178s
* Epoch 140  loss 2.399852  correct 48  time 0.4064s
* Epoch 150  loss 1.084895  correct 47  time 0.3996s
* Epoch 160  loss 1.832289  correct 45  time 0.3954s
* Epoch 170  loss 0.510964  correct 47  time 0.3945s
* Epoch 180  loss 0.882859  correct 49  time 0.3872s
* Epoch 190  loss 2.453670  correct 49  time 0.3804s
* Epoch 200  loss 2.158438  correct 48  time 0.3743s
* Epoch 210  loss 2.793679  correct 45  time 0.3743s
* Epoch 220  loss 0.573692  correct 48  time 0.3691s
* Epoch 230  loss 0.921076  correct 49  time 0.3643s
* Epoch 240  loss 3.472764  correct 49  time 0.3599s
* Epoch 250  loss 0.576717  correct 48  time 0.3577s
* Epoch 260  loss 1.662308  correct 49  time 0.3564s
* Epoch 270  loss 3.395201  correct 49  time 0.3527s
* Epoch 280  loss 0.171300  correct 50  time 0.3495s
* Epoch 290  loss 0.165587  correct 46  time 0.3463s
* Epoch 300  loss 0.424230  correct 47  time 0.3471s
* Epoch 310  loss 1.176225  correct 50  time 0.3442s
* Epoch 320  loss 1.065632  correct 50  time 0.3415s
* Epoch 330  loss 1.477093  correct 48  time 0.3389s
* Epoch 340  loss 0.539552  correct 49  time 0.3395s
* Epoch 350  loss 0.912111  correct 50  time 0.3376s
* Epoch 360  loss 0.545434  correct 47  time 0.3354s
* Epoch 370  loss 0.956788  correct 49  time 0.3333s
* Epoch 380  loss 0.984185  correct 47  time 0.3319s
* Epoch 390  loss 1.565412  correct 47  time 0.3326s
* Epoch 400  loss 3.177514  correct 49  time 0.3307s
* Epoch 410  loss 0.660330  correct 49  time 0.3289s
* Epoch 420  loss 0.367859  correct 47  time 0.3273s
* Epoch 430  loss 1.921270  correct 47  time 0.3283s
* Epoch 440  loss 0.217269  correct 47  time 0.3267s
* Epoch 450  loss 1.070678  correct 47  time 0.3251s
* Epoch 460  loss 1.207797  correct 49  time 0.3237s
* Epoch 470  loss 0.936183  correct 50  time 0.3238s
* Epoch 480  loss 0.223950  correct 49  time 0.3232s
* Epoch 490  loss 0.017724  correct 49  time 0.3218s

### GPU Training

#### Hyperparameters

* Size of Hidden Layer: 200
* Learning Rate: 0.05
* Number of Epochs: 490

#### Output Record

* Training Time per Epoch: 1.7965s
* Final Accuracy: 50/50 = 1.0

#### Output Training Log

* Epoch 0  loss 17.539318  correct 35  time 5.5224s
* Epoch 10  loss 3.191693  correct 50  time 2.0924s
* Epoch 20  loss 1.765595  correct 48  time 1.9729s
* Epoch 30  loss 1.806578  correct 50  time 1.9033s
* Epoch 40  loss 0.857782  correct 49  time 1.8768s
* Epoch 50  loss 0.727686  correct 49  time 1.8594s
* Epoch 60  loss 1.340661  correct 49  time 1.8404s
* Epoch 70  loss 1.022018  correct 50  time 1.8433s
* Epoch 80  loss 1.346959  correct 50  time 1.8424s
* Epoch 90  loss 0.951409  correct 50  time 1.8399s
* Epoch 100  loss 0.514679  correct 49  time 1.8307s
* Epoch 110  loss 0.915346  correct 50  time 1.8310s
* Epoch 120  loss 0.349927  correct 50  time 1.8241s
* Epoch 130  loss 0.407542  correct 50  time 1.8252s
* Epoch 140  loss 0.239649  correct 50  time 1.8193s
* Epoch 150  loss 0.122717  correct 50  time 1.8204s
* Epoch 160  loss 0.446936  correct 50  time 1.8157s
* Epoch 170  loss 0.346698  correct 50  time 1.8129s
* Epoch 180  loss 0.346247  correct 50  time 1.8129s
* Epoch 190  loss 0.379571  correct 50  time 1.8098s
* Epoch 200  loss 0.320212  correct 50  time 1.8113s
* Epoch 210  loss 0.562952  correct 50  time 1.8081s
* Epoch 220  loss 0.638630  correct 50  time 1.8126s
* Epoch 230  loss 0.283472  correct 50  time 1.8094s
* Epoch 240  loss 0.600253  correct 50  time 1.8097s
* Epoch 250  loss 0.226458  correct 50  time 1.8075s
* Epoch 260  loss 0.097700  correct 50  time 1.8074s
* Epoch 270  loss 0.147321  correct 50  time 1.8055s
* Epoch 280  loss 0.272603  correct 50  time 1.8032s
* Epoch 290  loss 0.025938  correct 50  time 1.8040s
* Epoch 300  loss 0.068906  correct 50  time 1.8027s
* Epoch 310  loss 0.513463  correct 50  time 1.8033s
* Epoch 320  loss 0.248661  correct 50  time 1.8014s
* Epoch 330  loss 0.194452  correct 50  time 1.8021s
* Epoch 340  loss 0.364202  correct 50  time 1.8004s
* Epoch 350  loss 0.394795  correct 50  time 1.8009s
* Epoch 360  loss 0.148550  correct 50  time 1.8016s
* Epoch 370  loss 0.356558  correct 50  time 1.8015s
* Epoch 380  loss 0.447911  correct 50  time 1.8008s
* Epoch 390  loss 0.095773  correct 50  time 1.7993s
* Epoch 400  loss 0.309203  correct 50  time 1.7996s
* Epoch 410  loss 0.032759  correct 50  time 1.7980s
* Epoch 420  loss 0.017313  correct 50  time 1.7988s
* Epoch 430  loss 0.036351  correct 50  time 1.7973s
* Epoch 440  loss 0.091099  correct 50  time 1.7980s
* Epoch 450  loss 0.352784  correct 50  time 1.7969s
* Epoch 460  loss 0.073364  correct 50  time 1.7964s
* Epoch 470  loss 0.097836  correct 50  time 1.7968s
* Epoch 480  loss 0.172779  correct 50  time 1.7958s
* Epoch 490  loss 0.178023  correct 50  time 1.7965s