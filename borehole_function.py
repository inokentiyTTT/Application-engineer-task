"""
Use case : Ishigami test function
=================================
"""
import openturns as ot
import math as m


class BoreholeModel:
    """
    Data class for the Ishigami model.


    Attributes
    ----------


    dim : The dimension of the problem
          dim = 3

    a : Constant
        a = 7.0

    b : Constant
        b = 0.1

    X1 : `Uniform` distribution
         First marginal, ot.Uniform(-np.pi, np.pi)

    X2 : `Uniform` distribution
         Second marginal, ot.Uniform(-np.pi, np.pi)

    X3 : `Uniform` distribution
         Third marginal, ot.Uniform(-np.pi, np.pi)

    distributionX : `ComposedDistribution`
                    The joint distribution of the input parameters.

    ishigami : `SymbolicFunction`
               The Ishigami model with a, b as variables.

    model : `ParametricFunction`
            The Ishigami model with the a=7.0 and b=0.1 parameters fixed.

    expectation : Constant
                  Expectation of the output variable.

    variance : Constant
               Variance of the output variable.

    S1 : Constant
         First order Sobol index number 1

    S2 : Constant
         First order Sobol index number 2

    S3 : Constant
         First order Sobol index number 3

    S12 : Constant
          Second order Sobol index for marginals 1 and 2.

    S13 : Constant
          Second order Sobol index for marginals 1 and 3.

    S23 : Constant
          Second order Sobol index for marginals 2 and 3.

    S123 : Constant

    ST1 : Constant
          Total order Sobol index number 1.

    ST2 : Constant
          Total order Sobol index number 2.

    ST3 : Constant
          Total order Sobol index number 3.


    Examples
    --------
    >>> from openturns.usecases import ishigami_function
    >>> # Load the Ishigami model
    >>> im = ishigami_function.IshigamiModel()
    """

    def __init__(self):
        # dimension
        self.dim = 8
        # Fixed parameters for the Ishigami function

        # First marginal : rw
        self.rw = ot.Normal(0.1, 0.0161812)
        self.rw.setName("rw")
        self.rw.setDescription(["rw"])

        # Second marginal : r
        self.r = ot.LogNormal(7.71, 1.0056)
        self.r.setName("r")
        self.r.setDescription(["r"])

        # Third marginal : Tu
        self.Tu = ot.Uniform(63070.0, 115600.0)
        self.Tu.setName("Tu")
        self.Tu.setDescription(["Tu"])

        # Fourth marginal : Hu
        self.Hu = ot.Uniform(990.0, 1110.0)
        self.Hu.setName("Hu")
        self.Hu.setDescription(["Hu"])

        # Fifth marginal : Tl
        self.Tl = ot.Uniform(63.1, 116.0)
        self.Tl.setName("Tl")
        self.Tl.setDescription(["Tl"])

        # Sixth : Hl
        self.Hl = ot.Uniform(700.0, 820.0)
        self.Hl.setName("Hl")
        self.Hl.setDescription(["Hl"])

        # Seventh : L
        self.L = ot.Uniform(1120.0, 1680.0)
        self.L.setName("L")
        self.L.setDescription(["L"])

        # Eighth : Kw
        self.Kw = ot.Uniform(9855.0, 12045.0)
        self.Kw.setName("Kw")
        self.Kw.setDescription(["Kw"])

        # Input distribution
        self.distributionX = ot.ComposedDistribution([self.rw, self.r, self.Tu, self.Hu, self.Tl, self.Hl, self.L, self.Kw])
        self.distributionX.setDescription(["rw", "r", "Tu", "Hu", "Tl", "Hl", "L", "Kw"])

        self.borehole = ot.SymbolicFunction(
            # ["X1", "X2", "X3", "a", "b"],
            ["rw", "r", "Tu", "Hu", "Tl", "Hl", "L", "Kw"],

            ["(2*pi_*Tu*(Hu-Hl))/(ln(r/rw)*(1+(2*L*Tu)/(ln(r/rw)*rw^2*Kw)+Tu/Tl))"],
        )
        self.borehole.setOutputDescription(["y"])
        
        self.model = self.borehole
        # The Ishigami model
        # self.model = ot.ParametricFunction(self.borehole,[3, 4], False)
        # self.model = ot.ParametricFunction(self.borehole, [0,1,2,3,4,5,6,7,8])

        # self.expectation = self.a / 2.0
        # self.variance = (
        #     1.0 / 2
        #     + self.a ** 2 / 8.0
        #     + self.b * m.pi ** 4 / 5.0
        #     + self.b ** 2 * m.pi ** 8 / 18.0
        # )
        # self.S1 = (
        #     1.0 / 2.0 + self.b * m.pi ** 4 / 5.0 + self.b ** 2 * m.pi ** 8 / 50.0
        # ) / self.variance
        # self.S2 = (self.a ** 2 / 8.0) / self.variance
        # self.S3 = 0.0
        # self.S12 = 0.0
        # self.S13 = (
        #     self.b ** 2 * m.pi ** 8 / 2.0 * (1.0 / 9.0 - 1.0 / 25.0) / self.variance
        # )
        # self.S23 = 0.0
        # self.S123 = 0.0
        # self.ST1 = self.S1 + self.S13
        # self.ST2 = self.S2
        # self.ST3 = self.S3 + self.S13
