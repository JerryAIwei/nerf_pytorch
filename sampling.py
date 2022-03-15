from setup import *
def stratified_sampling(t_n, t_f, N):
    diff = t_f - t_n
    randoms = np.random.uniform(0, diff/N, N)
    results = np.linspace(t_n, t_f, N, dtype = np.float32)
    results += randoms
    return results


def pdf_sampling(bins, weights, N_samples, det=False):
    # Get pdf
    weights += 1e-5  # prevent nans
    pdf = weights / np.reduce_sum(weights, -1, keepdims=True)
    cdf = np.cumsum(pdf, -1)
    cdf = np.concat([np.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = np.linspace(0., 1., N_samples)
        u = np.broadcast_to(u, list(cdf.shape[:-1]) + [N_samples])
    else:
        u = np.random.uniform(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    inds = np.searchsorted(cdf, u, side='right')
    below = np.maximum(0, inds-1)
    above = np.minimum(cdf.shape[-1]-1, inds)
    inds_g = np.stack([below, above], -1)
    cdf_g = np.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    bins_g = np.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = np.where(denom < 1e-5, np.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples