

def kl_linear(epzero, epmax, iter_per_epoch, curr_step, maxval=1.):
    number_zero, original_zero = epzero, epmax
    max_zero_step, original_anneal = number_zero * iter_per_epoch, original_zero * iter_per_epoch
    beta_t = max((maxval/(original_anneal - max_zero_step)) * (curr_step - max_zero_step), 0.)
    return min(maxval, beta_t)
