import itertools
from numpy import *
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.interpolate import interp1d

class fisher(object):
    """
    Base class to perform a Fisher Analysis from specified cosmology,
    survey and cosmological parameters.
    """

    def __init__(self, Fisher, params, fid_cosmo, margin_params=[]):
        """
        Constructor
        """
        self.params = params

        # Checks that the marginalisation parameters are actually considered
        # in the Fisher analysis
        self.margin_params = margin_params

        self.fid_cosmo = fid_cosmo

        # Precomputed Fisher matrix
        self._fullMat = Fisher
        self._fullInvMat = None
        self._mat = None
        self._invmat = None

    def Fij(self, param_i, param_j):
        """
            Returns the matrix element of the Fisher matrix for parameters
            param_i and param_j
        """
        i = self.params.index(param_i)
        j = self.params.index(param_j)

        return self.mat[i, j]

    def invFij(self, param_i, param_j):
        """
            Returns the matrix element of the inverse Fisher matrix for
            parameters param_i and param_j
        """
        i = self.params.index(param_i)
        j = self.params.index(param_j)

        return self.invmat[i, j]

    def sigma_fix(self, param):
        return 1.0 / sqrt(self.Fij(param, param))

    def sigma_marg(self, param):
        return sqrt(self.invFij(param, param))


    def _marginalise(self, params):
        r""" Marginalises the Fisher matrix over unwanted parameters.
        Parameters
        ----------
        params: list
            List of parameters that should not be marginalised over.
        Returns
        -------
        (mat, invmat): ndarray
            Marginalised Fisher matrix and its invers
        """
        # Builds inverse matrix
        marg_inv = zeros((len(params), len(params)))
        for i in range(len(params)):
            indi = self.params.index(params[i])
            for j in range(len(params)):
                indj = self.params.index(params[j])
                marg_inv[i, j] = self.invmat[indi, indj]

        marg_mat = linalg.pinv(marg_inv)

        return (marg_mat, marg_inv)

    def corner_plot(self, nstd=2, labels=None, **kwargs):
        r""" Makes a corner plot including all the parameters in the Fisher analysis
        """

        if labels is None:
            labels = self.params

        for i in range(len(self.params)):
            for j in range(i):
                ax = plt.subplot(len(self.params)-1, len(self.params)-1 , (i - 1)*(len(self.params)-1) + (j+1))
                if i == len(self.params) - 1:
                    ax.set_xlabel(labels[j])
                else:
                    ax.set_xticklabels([])
                if j == 0:
                    ax.set_ylabel(labels[i])
                else:
                    ax.set_yticklabels([])

                self.plot(self.params[j], self.params[i], nstd=nstd, ax=ax, **kwargs)

        plt.subplots_adjust(wspace=0)
        plt.subplots_adjust(hspace=0)

    def plot(self, p1, p2, nstd=2, ax=None, **kwargs):
        r""" Plots confidence contours corresponding to the parameters
        provided.
        Parameters
        ----------
        """
        params = [p1, p2]

        def eigsorted(cov):
            vals, vecs = linalg.eigh(cov)
            order = vals.argsort()[::-1]
            return vals[order], vecs[:, order]

        mat, cov = self._marginalise(params)
        # First find the fiducial value for the parameter in question
        fid_param = None
        pos = [0, 0]
        for p in params:
            fid_param = self.fid_cosmo[p]
            pos[params.index(p)] = fid_param

        if ax is None:
            ax = plt.gca()

        vals, vecs = eigsorted(cov)
        theta = degrees(arctan2(*vecs[:, 0][::-1]))

        # Width and height are "full" widths, not radius
        width, height = 2 * nstd * sqrt(vals)
        ellip = Ellipse(xy=pos, width=width,
                        height=height, angle=theta, **kwargs)

        ax.add_artist(ellip)
        sz = max(width, height)
        s1 = 1.5*nstd*self.sigma_marg(p1)
        s2 = 1.5*nstd*self.sigma_marg(p2)
        ax.set_xlim(pos[0] - s1, pos[0] + s1)
        ax.set_ylim(pos[1] - s2, pos[1] + s2)
        #ax.set_xlim(pos[0] - sz, pos[0] + sz)
        #ax.set_ylim(pos[1] - sz, pos[1] + sz)
        plt.draw()
        return ellip

    @property
    def FoM_DETF(self):
        """
            Computes the figure of merit from the Dark Energy Task Force
            Albrecht et al 2006
            FoM = 1/sqrt(det(F^-1_{w0,wa}))
        """
        det = (self.invFij('w0', 'w0') * self.invFij('wa', 'wa') -
               self.invFij('wa', 'w0') * self.invFij('w0', 'wa'))
        return 1.0 / sqrt(det)

    @property
    def FoM(self):
        """
            Total figure of merit : ln (1/det(F^{-1}))
        """
        return log(1.0 / abs(linalg.det(self.invmat)))

    @property
    def invmat(self):
        """
        Returns the inverse fisher matrix
        """
        if self._invmat is None:
            self._invmat = linalg.inv(self.mat)
        return self._invmat

    @property
    def mat(self):
        """
        Returns the fisher matrix marginalised over nuisance parameters
        """
        # If the matrix is not already computed, compute it
        if self._mat is None:
            self._fullInvMat = linalg.pinv(self._fullMat)

            # Apply marginalisation over nuisance parameters
            self._invmat = self._fullInvMat[0:len(self.params),
                                           0:len(self.params)]

            self._mat = linalg.pinv(self._invmat)
        return self._mat