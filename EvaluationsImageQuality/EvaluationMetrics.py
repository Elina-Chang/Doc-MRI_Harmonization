import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

from Tools.EarthMoverDistance import get_relative_frequencies, earth_mover_distance


def cal_psnr(im1, im2):
    mse = (np.abs(im1 - im2) ** 2).mean()
    psnr = 10 * np.log10(1.0 / mse)
    psnr = format(psnr, ".4f")
    return psnr


def cal_ssim(im1, im2):
    assert len(im1.shape) == 2 and len(im2.shape) == 2
    assert im1.shape == im2.shape
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, 255
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2
    l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    ssim = l12 * c12 * s12
    ssim = format(ssim, ".4f")
    return ssim


def calculate_emd(pixels_list1, pixels_list2):
    f1 = get_relative_frequencies(pixels_list1)
    f2 = get_relative_frequencies(pixels_list2)
    emd = stats.wasserstein_distance(f1, f2)
    emd = format(emd)
    return emd


# Calculate KL divergence
def caculate_kl(img1, img2):
    n1, bins1, patches1 = plt.hist(img1, bins=99)
    n2, bins2, patches2 = plt.hist(img2, bins=99)
    n1 = n1 / n1.sum()  # frequency ——> relative frequency
    n2 = n2 / n2.sum()
    kl = stats.entropy(n1, n2)
    kl = format(kl, ".2f")
    return kl


# Calculate frechet inception distance
def calculate_fid(act1, act2):
    import numpy
    from numpy import cov
    from numpy import iscomplexobj
    from numpy import trace
    from scipy.linalg import sqrtm
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    fid = format(fid, ".2f")
    return fid
