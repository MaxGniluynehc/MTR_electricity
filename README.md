
## MTR_electricity

- Pilot studies use a subset of the raw data. 
  - `pilot1`, `pilot3` and `pilot3`: take the 3 **key features** [intercept, slope, step_size] of 4 cycles to predict 
  those of the next cycle. If the seq_len = 5 of the batches, then the input dim is
  [4, batch_size, 3] and the output dim is [batch_size, 3]. 

  - `pilot4`: take [intercept, slope, step_size] of 4 cycles. The 5th cycle is divided into 
  Nsubintervals=3 subintervals, which gives Nsubintervals+1=4 subgroups --- a null group with no observation from 
  the 5th cycle, a group with 1/Nsubintervals observations from the 5th cycle, ..., and a group with full 
  observations from the 5th cycle. If the seq_len = 5 of the batches, then the input dim is
  [5, batch_size, 3] for each sub_batches and we have Nsubintervals+1=4 sub_batches.

  - `pilot5`: also take 3 more **peak features** [on_peak, idle, hour], indicating the rush hours of each cycle. The
  new features are concatenated to the **key features**. The input dim is [4, batch_size, 6] and the output dim is 
  still [batch_size, 3]. The model maps the input to [4, batch_size, 3], then pass it through (lstm_enc, att, lstm_dec).

  - `pilot6`:  also take 3 more **peak features** [on_peak, idle, hour], but are not concatenated. The inputs are
  (x,y), where x_dim = ydim = [4, batch_size, 3]. x and y are separately passed through two Linear layers, then 
  added together to get x_in, with dim [4, batch_size, in_size], then passed through (Linear, lstm_enc, 
  att, lstm_dec). That is, 
     - x_in = fc_inx1(x) + fc_iny1(y)
     - output = (fc_x2, lstm_enc, att, lstm_dec)(x_in)
  






