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
    ```                                                                                          |  
    assert a_shape[-1] == b_shape[-2]                                                            |  
    ```                                                                                          |  
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