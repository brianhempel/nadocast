module NNTrain

import Flux

# Mutates model_params via the optimizer.
#
# get_next_chunk should return [(X1, Y1), (X2, Y2), ...] where X1,Y1 is minibatch 1 etc...
#
# For 2D data, X1 etc should be 4-dimensional (x, y, channels, image in batch) or (y, x, channels, image in batch)
#
# Stops when get_next_chunk returns []
function train_one_epoch!(get_next_chunk, model_loss, optimizer)
  data_chunk = get_next_chunk()
  while !isempty(data_chunk)
    Flux.train!(model_loss, data_chunk, optimizer)
    data_chunk = get_next_chunk()
  end
end

end # module NNTrain

