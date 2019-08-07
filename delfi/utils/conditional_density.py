import numpy as np
from copy import deepcopy

def eval_conditional_density(pdf, theta, dim1, dim2, resolution=20, log=True):

    gbar_dim1 = np.linspace(-np.sqrt(3), np.sqrt(3), resolution)
    gbar_dim2 = np.linspace(-np.sqrt(3), np.sqrt(3), resolution)

    p_image = np.zeros((resolution, resolution))

    for index_gbar1 in range(resolution):
        for index_gbar2 in range(resolution):
            current_point_eval = deepcopy(theta)[0]
            current_point_eval[dim1] = gbar_dim1[index_gbar1]
            current_point_eval[dim2] = gbar_dim2[index_gbar2]

            p = pdf.eval(current_point_eval)
            p_image[index_gbar1, index_gbar2] = p

    if log:
        return p_image
    else:
        return np.exp(p_image)


class conditional_correlation:
    def __init__(self, cPDF, lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y):
        self.lx = lower_bound_x
        self.ux = upper_bound_x
        self.ly = lower_bound_y
        self.uy = upper_bound_y
        self.resolution_y, self.resolution_x = np.shape(cPDF)
        self.pdfXY = self.normalize_pdf_2D(cPDF)
        self.pdfX  = None
        self.pdfY  = None
        self.EX    = None
        self.EY    = None
        self.EXY   = None
        self.VarX  = None
        self.VarY  = None
        self.CovXY = None
        self.rho   = None

    @staticmethod
    def normalize_pdf_1D(pdf, lower, upper, resolution):
        return pdf * resolution / (upper - lower) / np.sum(pdf)

    def normalize_pdf_2D(self, pdf):
        return pdf * self.resolution_x * self.resolution_y / (self.ux - self.lx) /\
               (self.uy - self.ly) / np.sum(pdf)

    def calc_rhoXY(self):
        self.calc_marginals()
        self.calc_EXY()
        self.EX  = conditional_correlation.calc_E_1D(self.pdfX, self.lx, self.ux, self.resolution_x)
        self.EY = conditional_correlation.calc_E_1D(self.pdfY, self.ly, self.uy, self.resolution_y)
        self.CovXY = self.EXY - self.EX * self.EY
        self.VarX = conditional_correlation.calc_var1D(self.pdfX, self.lx, self.ux, self.resolution_x)
        self.VarY = conditional_correlation.calc_var1D(self.pdfY, self.ly, self.uy, self.resolution_y)
        self.rho = self.CovXY / np.sqrt(self.VarX * self.VarY)

    def calc_EXY(self):
        x_matrix = np.tile(np.linspace(self.lx, self.ux, self.resolution_x), (self.resolution_y, 1))
        y_matrix = np.tile(np.linspace(self.ly, self.uy, self.resolution_y), (self.resolution_x, 1)).T
        self.EXY = np.sum(  np.sum(  x_matrix * y_matrix * self.pdfXY  )  )
        self.EXY /= self.resolution_x * self.resolution_y / (self.ux - self.lx) / (self.uy - self.ly)

    @staticmethod
    def calc_E_1D(pdf, lower, upper, resolution):
        x_vector = np.linspace(lower, upper, resolution)
        E = np.sum(x_vector * pdf)
        E /= resolution / (upper - lower)
        return E

    @staticmethod
    def calc_var1D(pdf, lower, upper, resolution):
        x_vector = np.linspace(lower, upper, resolution)
        E2 = np.sum(x_vector**2 * pdf)
        E2 /= resolution / (upper - lower)
        var = E2 - conditional_correlation.calc_E_1D(pdf, lower, upper, resolution)**2
        return var

    def calc_marginals(self):
        self.pdfX = np.sum(self.pdfXY, axis=0)
        self.pdfY = np.sum(self.pdfXY, axis=1)

        self.pdfX = conditional_correlation.normalize_pdf_1D(self.pdfX, self.lx, self.ux, self.resolution_x)
        self.pdfY = conditional_correlation.normalize_pdf_1D(self.pdfY, self.ly, self.uy, self.resolution_y)
