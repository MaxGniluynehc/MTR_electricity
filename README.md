
## MTR_electricity

- Pilot studies use a subset of the raw data. 
  - `pilot1`, `pilot3` and `pilot3`: take [intercept, slope, step_size] of 4 cycles to predict 
  those of the next cycle. If the seq_len = 5 of the batches, then the input dim is
  [4, batch_size, 3] and the output is [batch_size, 3]. 

  - `pilot4`: take [intercept, slope, step_size] of 4 cycles. The 5th cycle is divided into 
  Nsubintervals=3 subintervals, which gives Nsubintervals+1=4 subgroups --- a null group with no observation from 
  the 5th cycle, a group with 1/Nsubintervals observations from the 5th cycle, ..., and a group with full 
  observations from the 5th cycle. If the seq_len = 5 of the batches, then the input dim is
  [5, batch_size, 3] for each sub_batches and we have Nsubintervals+1=4 sub_batches. 






